#!/usr/bin/env python
"""Discovery-eligible atlas FDR audit for Phase 12.

The full expanded atlas contains many weakly annotated "other/background" LR
pairs. The strict pooled audit keeps those pairs in one BH family and remains
the broadest FDR check. This script adds a narrower, label-independent audit:
test LR pairs that have a functional category annotation and expression support.

Eligibility uses only LR metadata and observed expression support:
- biological_category != "other";
- n_nonzero_expr >= 3;
- expr_gate >= 0.5;
- both ligand and receptor are observed in at least one candidate edge.

Known high-expression controls and receptor/ligand-shared controls are retained
inside the eligible universe. This is not a positive-panel audit and does not
use true_lr to define eligibility.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
EXPANDED_DIR = PHASE12_DIR / "atlas_model_expanded"
PERM_DIR = EXPANDED_DIR / "permutation_audit"
MATCHED_DIR = PHASE12_DIR / "matched_spatial_atlas_fdr"
OUT_DIR = PHASE12_DIR / "discovery_eligible_atlas_fdr"
TOP_K_VALUES = (5, 10, 20, 50, 100)
Q_THRESHOLD = 0.10


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ranking = pd.read_csv(EXPANDED_DIR / "phase12_atlas_model_expanded_calibrated_lr_ranking.csv")
    permutation = pd.read_csv(PERM_DIR / "phase12_expanded_atlas_permutation_lr_fdr.csv")
    matched = pd.read_csv(MATCHED_DIR / "phase12_matched_spatial_atlas_fdr_lr.csv")

    table = build_table(ranking, permutation, matched)
    eligible = table[table["discovery_eligible"].astype(bool)].copy()
    topk = summarize_topk(eligible)
    dataset = summarize_dataset(table, eligible, topk)
    story = summarize_story(dataset)

    table.to_csv(OUT_DIR / "phase12_discovery_eligible_atlas_all_lr.csv", index=False)
    eligible.to_csv(OUT_DIR / "phase12_discovery_eligible_atlas_lr.csv", index=False)
    topk.to_csv(OUT_DIR / "phase12_discovery_eligible_atlas_topk.csv", index=False)
    dataset.to_csv(OUT_DIR / "phase12_discovery_eligible_atlas_dataset.csv", index=False)
    story.to_csv(OUT_DIR / "phase12_discovery_eligible_atlas_story.csv", index=False)
    write_status(dataset, eligible, story)
    write_manifest()

    print(dataset.to_string(index=False))
    print()
    print(eligible.groupby("dataset_key").head(20).to_string(index=False))
    print()
    print(story.to_string(index=False))


def build_table(ranking: pd.DataFrame, permutation: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    keep_perm = [
        "dataset_key",
        "lr_id",
        "conservative_global_empirical_p",
        "strict_conservative_global_empirical_p",
        "receiver_permutation_global_empirical_p",
        "lr_identity_permutation_global_empirical_p",
        "spatial_distance_permutation_global_empirical_p",
    ]
    keep_matched = [
        "dataset_key",
        "lr_id",
        "matched_spatial_empirical_p",
        "matched_all_null_empirical_p",
        "strict_matched_all_null_empirical_p",
    ]
    out = ranking.merge(permutation[keep_perm], on=["dataset_key", "lr_id"], how="left")
    out = out.merge(matched[keep_matched], on=["dataset_key", "lr_id"], how="left")
    out = annotate_controls(out)
    out["discovery_eligible"] = (
        out["biological_category"].astype(str).ne("other")
        & pd.to_numeric(out["n_nonzero_expr"], errors="coerce").fillna(0).ge(3)
        & pd.to_numeric(out["expr_gate"], errors="coerce").fillna(0).ge(0.5)
        & pd.to_numeric(out["ligand_positive_fraction"], errors="coerce").fillna(0).gt(0)
        & pd.to_numeric(out["receptor_positive_fraction"], errors="coerce").fillna(0).gt(0)
    )
    out["discovery_eligibility_reason"] = np.where(
        out["discovery_eligible"],
        "functional_category_and_expression_supported",
        "not_discovery_eligible_background_or_low_support",
    )

    rows = []
    for _, sub in out.groupby("dataset_key", sort=False):
        sub = sub.copy()
        eligible_mask = sub["discovery_eligible"].astype(bool)
        for p_col, q_col in [
            ("conservative_global_empirical_p", "eligible_pooled_all_null_bh_fdr"),
            ("strict_conservative_global_empirical_p", "eligible_strict_pooled_all_null_bh_fdr"),
            ("matched_all_null_empirical_p", "eligible_matched_all_null_bh_fdr"),
            ("strict_matched_all_null_empirical_p", "eligible_strict_matched_all_null_bh_fdr"),
            ("matched_spatial_empirical_p", "eligible_matched_spatial_bh_fdr"),
        ]:
            sub[q_col] = np.nan
            idx = sub.index[eligible_mask]
            sub.loc[idx, q_col] = bh_adjust(pd.to_numeric(sub.loc[idx, p_col], errors="coerce").to_numpy(dtype=float))
        rows.append(sub)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out["eligible_fdr_call"] = [classify_lr(row) for row in out.itertuples(index=False)]
    return out.sort_values(["dataset_key", "rank", "lr_pair"]).reset_index(drop=True)


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


def summarize_topk(eligible: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_key, sub in eligible.groupby("dataset_key", sort=False):
        sub = sub.sort_values("rank")
        for top_k in TOP_K_VALUES:
            top = sub.head(top_k)
            if top.empty:
                continue
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "top_k": top_k,
                    "n_lr": int(len(top)),
                    "n_true_lr": int(top["true_lr"].sum()),
                    "n_primary_control_decoy": int(top["primary_control_decoy"].sum()),
                    "n_high_expression_control": int(top["false_positive_high_expression"].sum()),
                    "n_unlabeled_background": int(top["decoy_class"].eq("unlabeled_background_lr").sum()),
                    "n_eligible_pooled_q_0_1": int(top["eligible_pooled_all_null_bh_fdr"].le(Q_THRESHOLD).sum()),
                    "n_eligible_matched_q_0_1": int(top["eligible_matched_all_null_bh_fdr"].le(Q_THRESHOLD).sum()),
                    "n_eligible_strict_matched_q_0_1": int(
                        top["eligible_strict_matched_all_null_bh_fdr"].le(Q_THRESHOLD).sum()
                    ),
                    "top_lr_pairs": ";".join(top["lr_pair"].astype(str).head(12)),
                }
            )
    return pd.DataFrame(rows)


def summarize_dataset(all_lr: pd.DataFrame, eligible: pd.DataFrame, topk: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_key, all_sub in all_lr.groupby("dataset_key", sort=False):
        sub = eligible[eligible["dataset_key"].eq(dataset_key)].copy()
        top10 = topk[(topk["dataset_key"].eq(dataset_key)) & (topk["top_k"].eq(10))]
        t10 = top10.iloc[0].to_dict() if len(top10) else {}
        pooled_hits = sub[sub["eligible_pooled_all_null_bh_fdr"].le(Q_THRESHOLD)]
        matched_hits = sub[sub["eligible_matched_all_null_bh_fdr"].le(Q_THRESHOLD)]
        strict_matched_hits = sub[sub["eligible_strict_matched_all_null_bh_fdr"].le(Q_THRESHOLD)]
        best_true = sub[sub["true_lr"]].sort_values(["eligible_matched_all_null_bh_fdr", "rank"]).head(1)
        rows.append(
            {
                "dataset_key": dataset_key,
                "n_total_lr": int(len(all_sub)),
                "n_eligible_lr": int(len(sub)),
                "n_eligible_true_lr": int(sub["true_lr"].sum()),
                "n_eligible_primary_control_decoy": int(sub["primary_control_decoy"].sum()),
                "n_eligible_high_expression_control": int(sub["false_positive_high_expression"].sum()),
                "n_eligible_unlabeled_background": int(sub["decoy_class"].eq("unlabeled_background_lr").sum()),
                "top10_true_lr": int(t10.get("n_true_lr", 0)),
                "top10_primary_control_decoy": int(t10.get("n_primary_control_decoy", 0)),
                "top10_high_expression_control": int(t10.get("n_high_expression_control", 0)),
                "eligible_pooled_q_0_1": int(len(pooled_hits)),
                "eligible_pooled_true_q_0_1": int(pooled_hits["true_lr"].sum()) if len(pooled_hits) else 0,
                "eligible_pooled_control_q_0_1": int(pooled_hits["primary_control_decoy"].sum()) if len(pooled_hits) else 0,
                "eligible_pooled_background_q_0_1": int(
                    pooled_hits["decoy_class"].eq("unlabeled_background_lr").sum()
                )
                if len(pooled_hits)
                else 0,
                "eligible_matched_q_0_1": int(len(matched_hits)),
                "eligible_matched_true_q_0_1": int(matched_hits["true_lr"].sum()) if len(matched_hits) else 0,
                "eligible_matched_control_q_0_1": int(matched_hits["primary_control_decoy"].sum()) if len(matched_hits) else 0,
                "eligible_matched_background_q_0_1": int(
                    matched_hits["decoy_class"].eq("unlabeled_background_lr").sum()
                )
                if len(matched_hits)
                else 0,
                "eligible_strict_matched_q_0_1": int(len(strict_matched_hits)),
                "eligible_strict_matched_true_q_0_1": int(strict_matched_hits["true_lr"].sum())
                if len(strict_matched_hits)
                else 0,
                "eligible_strict_matched_control_q_0_1": int(strict_matched_hits["primary_control_decoy"].sum())
                if len(strict_matched_hits)
                else 0,
                "best_true_lr": str(best_true.iloc[0]["lr_pair"]) if len(best_true) else "",
                "best_true_eligible_matched_q": float(best_true.iloc[0]["eligible_matched_all_null_bh_fdr"])
                if len(best_true)
                else np.nan,
                "discovery_eligible_call": classify_dataset(sub),
            }
        )
    return pd.DataFrame(rows).sort_values("dataset_key")


def summarize_story(dataset: pd.DataFrame) -> pd.DataFrame:
    if dataset.empty:
        return pd.DataFrame()
    pooled_true = int(dataset["eligible_pooled_true_q_0_1"].sum())
    matched_true = int(dataset["eligible_matched_true_q_0_1"].sum())
    strict_matched_true = int(dataset["eligible_strict_matched_true_q_0_1"].sum())
    control_hits = int(dataset["eligible_matched_control_q_0_1"].sum())
    if matched_true > 0 and control_hits == 0:
        call = "eligible_discovery_fdr_supportive_controls_clean"
    elif matched_true > 0:
        call = "eligible_discovery_fdr_supportive_with_control_caveat"
    else:
        call = "eligible_discovery_fdr_not_supportive"
    return pd.DataFrame(
        [
            {
                "n_datasets": int(len(dataset)),
                "n_eligible_pooled_true_q_0_1": pooled_true,
                "n_eligible_matched_true_q_0_1": matched_true,
                "n_eligible_strict_matched_true_q_0_1": strict_matched_true,
                "n_eligible_matched_control_q_0_1": control_hits,
                "overall_discovery_eligible_call": call,
                "claim_boundary": "This is FDR within a pre-defined functionally annotated and expression-supported LR universe. It does not replace the broader all-covered-LR pooled FDR, and unlabeled background LR are not native false positives.",
            }
        ]
    )


def classify_lr(row: object) -> str:
    if not bool(getattr(row, "discovery_eligible", False)):
        return "not_discovery_eligible"
    strict_q = to_float(getattr(row, "eligible_strict_matched_all_null_bh_fdr", np.nan))
    matched_q = to_float(getattr(row, "eligible_matched_all_null_bh_fdr", np.nan))
    pooled_q = to_float(getattr(row, "eligible_pooled_all_null_bh_fdr", np.nan))
    if np.isfinite(strict_q) and strict_q <= Q_THRESHOLD:
        return "strict_matched_eligible_fdr_supported"
    if np.isfinite(matched_q) and matched_q <= Q_THRESHOLD:
        return "matched_eligible_fdr_supported"
    if np.isfinite(pooled_q) and pooled_q <= Q_THRESHOLD:
        return "pooled_eligible_fdr_supported"
    return "eligible_not_fdr_supported"


def classify_dataset(sub: pd.DataFrame) -> str:
    matched_true = int((sub["true_lr"] & sub["eligible_matched_all_null_bh_fdr"].le(Q_THRESHOLD)).sum())
    strict_true = int((sub["true_lr"] & sub["eligible_strict_matched_all_null_bh_fdr"].le(Q_THRESHOLD)).sum())
    matched_controls = int((sub["primary_control_decoy"] & sub["eligible_matched_all_null_bh_fdr"].le(Q_THRESHOLD)).sum())
    top10 = sub.sort_values("rank").head(10)
    top10_controls = int(top10["primary_control_decoy"].sum()) if len(top10) else 0
    if strict_true >= 2 and matched_controls == 0 and top10_controls == 0:
        return "strict_eligible_fdr_supportive_controls_clean"
    if matched_true >= 2 and matched_controls == 0:
        return "eligible_fdr_supportive_controls_clean"
    if matched_true > 0:
        return "eligible_fdr_supportive_with_control_caveat"
    return "eligible_fdr_not_supportive"


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


def to_float(value: object) -> float:
    try:
        if value is None or pd.isna(value):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def fmt(value: object) -> str:
    val = to_float(value)
    if not np.isfinite(val):
        return "NA"
    if abs(val) < 0.001 and val != 0:
        return f"{val:.2e}"
    if abs(val) >= 10:
        return f"{val:.2f}"
    return f"{val:.3g}"


def write_status(dataset: pd.DataFrame, eligible: pd.DataFrame, story: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Discovery-Eligible Atlas FDR Audit",
        "",
        "## Scope",
        "",
        "This audit tests a pre-defined LR universe with functional category annotation and observed expression support. It keeps hard controls inside the universe and does not use true_lr to define eligibility.",
        "",
        "## Dataset Summary",
        "",
    ]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.discovery_eligible_call}`: eligible={int(row.n_eligible_lr)}/{int(row.n_total_lr)}, "
            f"top10 true={int(row.top10_true_lr)}, top10 controls={int(row.top10_primary_control_decoy)}, "
            f"matched q<=0.1 true={int(row.eligible_matched_true_q_0_1)}, controls={int(row.eligible_matched_control_q_0_1)}, "
            f"background={int(row.eligible_matched_background_q_0_1)}, best true={row.best_true_lr} q={fmt(row.best_true_eligible_matched_q)}."
        )
    lines += ["", "## Top Eligible LR", ""]
    for row in eligible.groupby("dataset_key").head(12).itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` rank {int(row.rank)} `{row.lr_pair}`: {row.eligible_fdr_call}; "
            f"class={row.decoy_class}; matched q={fmt(row.eligible_matched_all_null_bh_fdr)}, "
            f"pooled q={fmt(row.eligible_pooled_all_null_bh_fdr)}."
        )
    lines += ["", "## Overall", ""]
    if not story.empty:
        s = story.iloc[0]
        lines.append(
            f"- `{s.overall_discovery_eligible_call}`: pooled true q<=0.1={int(s.n_eligible_pooled_true_q_0_1)}, "
            f"matched true q<=0.1={int(s.n_eligible_matched_true_q_0_1)}, "
            f"strict matched true q<=0.1={int(s.n_eligible_strict_matched_true_q_0_1)}, "
            f"matched controls q<=0.1={int(s.n_eligible_matched_control_q_0_1)}."
        )
    lines += [
        "",
        "## Claim Boundary",
        "",
        "- This supports discovery-level FDR only within the discovery-eligible universe.",
        "- Do not call unlabeled background LR false positives in real tissue.",
        "- Do not replace the broader all-covered-LR pooled FDR with this result.",
    ]
    (OUT_DIR / "phase12_discovery_eligible_atlas_fdr_status.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_manifest() -> None:
    manifest = {
        "task": "phase12_discovery_eligible_atlas_fdr",
        "eligibility_rule": {
            "biological_category": "not other",
            "n_nonzero_expr": ">=3",
            "expr_gate": ">=0.5",
            "ligand_positive_fraction": ">0",
            "receptor_positive_fraction": ">0",
        },
        "inputs": [
            "outputs/phase12/atlas_model_expanded/phase12_atlas_model_expanded_calibrated_lr_ranking.csv",
            "outputs/phase12/atlas_model_expanded/permutation_audit/phase12_expanded_atlas_permutation_lr_fdr.csv",
            "outputs/phase12/matched_spatial_atlas_fdr/phase12_matched_spatial_atlas_fdr_lr.csv",
        ],
        "outputs": [
            "outputs/phase12/discovery_eligible_atlas_fdr/phase12_discovery_eligible_atlas_all_lr.csv",
            "outputs/phase12/discovery_eligible_atlas_fdr/phase12_discovery_eligible_atlas_lr.csv",
            "outputs/phase12/discovery_eligible_atlas_fdr/phase12_discovery_eligible_atlas_topk.csv",
            "outputs/phase12/discovery_eligible_atlas_fdr/phase12_discovery_eligible_atlas_dataset.csv",
            "outputs/phase12/discovery_eligible_atlas_fdr/phase12_discovery_eligible_atlas_story.csv",
        ],
        "claim_boundary": "FDR within a pre-defined discovery-eligible LR universe, not the broader all-covered-LR pooled FDR.",
    }
    (OUT_DIR / "phase12_discovery_eligible_atlas_fdr_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
