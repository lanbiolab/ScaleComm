#!/usr/bin/env python
"""Strict discovery audit for Phase 12 expanded whole-atlas ranking.

This audit uses the stricter discovery score added to the calibrated atlas
ranking. It is meant as a conservative follow-up to the default calibrated
ranking: low annotation support, weak expression support, zero-expression LR and
hard-negative LR are penalized more strongly.

The output keeps the claim boundary explicit. A clean top-k target-decoy profile
supports prioritization, but a positive whole-atlas discovery claim still
requires strict permutation q-values.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
EXPANDED_DIR = PHASE12_DIR / "atlas_model_expanded"
PERM_DIR = EXPANDED_DIR / "permutation_audit"
OUT_DIR = PHASE12_DIR / "strict_atlas_discovery_audit"
TOP_K_VALUES = (5, 10, 20, 50, 100, 200)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ranking = pd.read_csv(EXPANDED_DIR / "phase12_atlas_model_expanded_calibrated_lr_ranking.csv")
    perm = read_csv(PERM_DIR / "phase12_expanded_atlas_permutation_lr_fdr.csv")
    if "strict_discovery_score" not in ranking.columns:
        raise ValueError("Calibrated ranking lacks strict_discovery_score; rerun calibration first.")

    annotated = annotate_decoys(ranking)
    lr_table = add_strict_control_fdr(annotated, perm)
    topk = summarize_topk(lr_table)
    dataset = summarize_dataset(lr_table, topk)
    story = summarize_story(dataset)

    lr_table.to_csv(OUT_DIR / "phase12_strict_atlas_discovery_lr.csv", index=False)
    topk.to_csv(OUT_DIR / "phase12_strict_atlas_discovery_topk.csv", index=False)
    dataset.to_csv(OUT_DIR / "phase12_strict_atlas_discovery_dataset.csv", index=False)
    story.to_csv(OUT_DIR / "phase12_strict_atlas_discovery_story.csv", index=False)
    write_status(dataset, topk, story)
    write_manifest(lr_table, topk, dataset, story)

    print(dataset.to_string(index=False))
    print()
    print(topk.to_string(index=False))
    print()
    print(lr_table.groupby("dataset_key").head(20).to_string(index=False))


def annotate_decoys(ranking: pd.DataFrame) -> pd.DataFrame:
    out = ranking.copy()
    out["true_lr"] = out["true_lr"].fillna(False).astype(bool)
    out["false_positive_high_expression"] = out["false_positive_high_expression"].fillna(False).astype(bool)
    out["n_nonzero_expr"] = pd.to_numeric(out["n_nonzero_expr"], errors="coerce").fillna(0)
    out["strict_rank"] = pd.to_numeric(out["strict_rank"], errors="coerce")
    out["strict_discovery_score"] = pd.to_numeric(out["strict_discovery_score"], errors="coerce")
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
    out["zero_expression_decoy"] = out["n_nonzero_expr"].le(0)
    out["primary_control_decoy"] = (
        out["false_positive_high_expression"]
        | out["receptor_shared_decoy"]
        | out["ligand_shared_decoy"]
    )
    out["strict_control_decoy"] = out["primary_control_decoy"] | out["zero_expression_decoy"]
    out["background_lr"] = out["selected_reason"].astype(str).eq("category_balanced")
    out["decoy_class"] = out.apply(decoy_class, axis=1)
    return out


def add_strict_control_fdr(table: pd.DataFrame, perm: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    perm_cols = [
        "dataset_key",
        "lr_id",
        "strict_conservative_global_empirical_p",
        "strict_conservative_global_bh_fdr",
        "strict_conservative_within_lr_empirical_p",
        "strict_permutation_support_class",
    ]
    if not perm.empty and all(col in perm.columns for col in perm_cols[:3]):
        table = table.merge(perm[[c for c in perm_cols if c in perm.columns]], on=["dataset_key", "lr_id"], how="left")
    else:
        table["strict_conservative_global_empirical_p"] = np.nan
        table["strict_conservative_global_bh_fdr"] = np.nan
        table["strict_conservative_within_lr_empirical_p"] = np.nan
        table["strict_permutation_support_class"] = "strict_permutation_not_available"

    for dataset_key, sub in table.groupby("dataset_key", sort=False):
        sub = sub.sort_values(["strict_rank", "lr_pair"]).reset_index(drop=True).copy()
        sub["strict_rank_position"] = np.arange(1, len(sub) + 1)
        for mode, column in [
            ("primary_control", "primary_control_decoy"),
            ("strict_control", "strict_control_decoy"),
        ]:
            decoy = sub[column].fillna(False).astype(bool).to_numpy()
            target = ~decoy
            cum_decoy = np.cumsum(decoy).astype(float)
            cum_target = np.cumsum(target).astype(float)
            prefix_fdr = (cum_decoy + 1.0) / np.maximum(cum_target, 1.0)
            sub[f"strict_rank_{mode}_cum_decoys"] = cum_decoy.astype(int)
            sub[f"strict_rank_{mode}_cum_targets"] = cum_target.astype(int)
            sub[f"strict_rank_{mode}_prefix_fdr"] = prefix_fdr
            sub[f"strict_rank_{mode}_target_decoy_q"] = np.minimum.accumulate(prefix_fdr[::-1])[::-1]
        out_rows.append(sub)
    return pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()


def summarize_topk(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_key, sub in table.groupby("dataset_key", sort=False):
        sub = sub.sort_values("strict_rank_position")
        for top_k in TOP_K_VALUES:
            top = sub.head(top_k)
            if top.empty:
                continue
            last = top.iloc[-1]
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "top_k": top_k,
                    "n_lr": int(len(top)),
                    "n_true_lr": int(top["true_lr"].sum()),
                    "n_background_lr": int(top["background_lr"].sum()),
                    "n_primary_control_decoy": int(top["primary_control_decoy"].sum()),
                    "n_strict_control_decoy": int(top["strict_control_decoy"].sum()),
                    "n_high_expression_decoy": int(top["false_positive_high_expression"].sum()),
                    "n_ligand_shared_decoy": int(top["ligand_shared_decoy"].sum()),
                    "n_receptor_shared_decoy": int(top["receptor_shared_decoy"].sum()),
                    "n_permutation_q_0_1": int(
                        pd.to_numeric(top["strict_conservative_global_bh_fdr"], errors="coerce").le(0.1).sum()
                    ),
                    "primary_control_prefix_fdr": float(last["strict_rank_primary_control_prefix_fdr"]),
                    "strict_control_prefix_fdr": float(last["strict_rank_strict_control_prefix_fdr"]),
                    "top_lr_pairs": ";".join(top["lr_pair"].astype(str).head(12)),
                }
            )
    return pd.DataFrame(rows)


def summarize_dataset(table: pd.DataFrame, topk: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_key, sub in table.groupby("dataset_key", sort=False):
        sub = sub.sort_values("strict_rank_position")
        top10 = pick(topk, dataset_key=dataset_key, top_k=10)
        top100 = pick(topk, dataset_key=dataset_key, top_k=100)
        q = pd.to_numeric(sub["strict_conservative_global_bh_fdr"], errors="coerce")
        true = sub["true_lr"].astype(bool)
        rows.append(
            {
                "dataset_key": dataset_key,
                "n_lr": int(len(sub)),
                "n_true_lr": int(true.sum()),
                "top10_true_lr": int(get(top10, "n_true_lr", 0)),
                "top10_background_lr": int(get(top10, "n_background_lr", 0)),
                "top10_primary_control_decoy": int(get(top10, "n_primary_control_decoy", 0)),
                "top10_high_expression_decoy": int(get(top10, "n_high_expression_decoy", 0)),
                "top10_primary_control_prefix_fdr": float(get(top10, "primary_control_prefix_fdr", np.nan)),
                "top100_primary_control_prefix_fdr": float(get(top100, "primary_control_prefix_fdr", np.nan)),
                "all_lr_strict_permutation_q_0_1": int(q.le(0.1).sum()),
                "all_true_lr_strict_permutation_q_0_1": int((true & q.le(0.1)).sum()),
                "best_true_lr": best_true_lr(sub),
                "best_true_lr_strict_fdr": best_true_lr_fdr(sub),
                "strict_atlas_discovery_call": classify_dataset(top10, top100, sub),
            }
        )
    return pd.DataFrame(rows)


def summarize_story(dataset: pd.DataFrame) -> pd.DataFrame:
    n_datasets = int(len(dataset))
    n_perm = int(pd.to_numeric(dataset["all_lr_strict_permutation_q_0_1"], errors="coerce").fillna(0).sum())
    n_top10_clean = int(pd.to_numeric(dataset["top10_primary_control_prefix_fdr"], errors="coerce").le(0.1).sum())
    n_top100_clean = int(pd.to_numeric(dataset["top100_primary_control_prefix_fdr"], errors="coerce").le(0.1).sum())
    if n_perm > 0:
        call = "strict_global_permutation_discovery_supported"
    elif n_top10_clean == n_datasets:
        call = "strict_top10_control_fdr_clean_permutation_blocked"
    elif n_top100_clean == n_datasets:
        call = "strict_top100_control_fdr_clean_permutation_blocked"
    else:
        call = "strict_audit_incomplete_or_not_clean"
    return pd.DataFrame(
        [
            {
                "n_datasets": n_datasets,
                "n_datasets_top10_primary_control_fdr_le_0_1": n_top10_clean,
                "n_datasets_top100_primary_control_fdr_le_0_1": n_top100_clean,
                "n_all_lr_strict_permutation_q_0_1": n_perm,
                "overall_strict_atlas_call": call,
                "claim_boundary": "Strict discovery ranking may improve top-k control contamination, but whole-atlas discovery FDR is supported only if strict permutation q<=0.1 discoveries are present.",
            }
        ]
    )


def classify_dataset(top10: dict[str, Any], top100: dict[str, Any], sub: pd.DataFrame) -> str:
    n_q = int(pd.to_numeric(sub["strict_conservative_global_bh_fdr"], errors="coerce").le(0.1).sum())
    top10_true = int(get(top10, "n_true_lr", 0))
    top10_primary = float(get(top10, "primary_control_prefix_fdr", np.nan))
    top100_primary = float(get(top100, "primary_control_prefix_fdr", np.nan))
    if n_q > 0 and top10_true > 0 and np.isfinite(top10_primary) and top10_primary <= 0.1:
        return "strict_permutation_and_control_fdr_supported"
    if n_q > 0:
        return "strict_permutation_supported_control_caveat"
    if top10_true > 0 and np.isfinite(top10_primary) and top10_primary <= 0.1:
        return "strict_top10_control_fdr_clean_permutation_blocked"
    if top10_true > 0 and np.isfinite(top100_primary) and top100_primary <= 0.1:
        return "strict_top100_control_fdr_clean_top10_caveat_permutation_blocked"
    return "strict_discovery_not_supported"


def decoy_class(row: pd.Series) -> str:
    if bool(row.get("false_positive_high_expression", False)):
        return "high_expression_control"
    if bool(row.get("receptor_shared_decoy", False)):
        return "receptor_shared_control"
    if bool(row.get("ligand_shared_decoy", False)):
        return "ligand_shared_control"
    if bool(row.get("zero_expression_decoy", False)):
        return "zero_expression_control"
    if bool(row.get("true_lr", False)):
        return "prespecified_positive_anchor"
    if bool(row.get("background_lr", False)):
        return "unlabeled_background_lr"
    return "unlabeled_lr"


def write_status(dataset: pd.DataFrame, topk: pd.DataFrame, story: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Strict Atlas Discovery Audit",
        "",
        "## Scope",
        "",
        "This audit ranks LR pairs by strict_discovery_score and reports hard-control contamination plus strict permutation FDR. It is stricter than the default calibrated ranking, but it is still a post-hoc audit layer.",
        "",
        "## Dataset Summary",
        "",
    ]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.strict_atlas_discovery_call}`: LR={int(row.n_lr)}, "
            f"top10 true={int(row.top10_true_lr)}, top10 background={int(row.top10_background_lr)}, "
            f"top10 primary decoy={int(row.top10_primary_control_decoy)}, "
            f"top10 primary FDR={fmt(row.top10_primary_control_prefix_fdr)}, "
            f"top100 primary FDR={fmt(row.top100_primary_control_prefix_fdr)}, "
            f"all-LR strict q<=0.1={int(row.all_lr_strict_permutation_q_0_1)}, "
            f"best true={row.best_true_lr} q={fmt(row.best_true_lr_strict_fdr)}."
        )
    lines += ["", "## Top-K Strict Ranking", ""]
    for row in topk[topk["top_k"].isin([10, 20, 50, 100])].itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` top{int(row.top_k)}: true={int(row.n_true_lr)}, "
            f"background={int(row.n_background_lr)}, primary decoy={int(row.n_primary_control_decoy)}, "
            f"q<=0.1={int(row.n_permutation_q_0_1)}, primary FDR={fmt(row.primary_control_prefix_fdr)}."
        )
    lines += ["", "## Overall", ""]
    if not story.empty:
        s = story.iloc[0]
        lines.append(
            f"- `{s.overall_strict_atlas_call}`: datasets={int(s.n_datasets)}, "
            f"top10-control-clean={int(s.n_datasets_top10_primary_control_fdr_le_0_1)}, "
            f"top100-control-clean={int(s.n_datasets_top100_primary_control_fdr_le_0_1)}, "
            f"all-LR strict permutation discoveries={int(s.n_all_lr_strict_permutation_q_0_1)}."
        )
        lines.append(f"- Claim boundary: {s.claim_boundary}")
    (OUT_DIR / "phase12_strict_atlas_discovery_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(lr_table: pd.DataFrame, topk: pd.DataFrame, dataset: pd.DataFrame, story: pd.DataFrame) -> None:
    manifest = {
        "task": "phase12_strict_atlas_discovery_audit",
        "n_lr_rows": int(len(lr_table)),
        "n_topk_rows": int(len(topk)),
        "n_dataset_rows": int(len(dataset)),
        "n_story_rows": int(len(story)),
        "inputs": [
            "outputs/phase12/atlas_model_expanded/phase12_atlas_model_expanded_calibrated_lr_ranking.csv",
            "outputs/phase12/atlas_model_expanded/permutation_audit/phase12_expanded_atlas_permutation_lr_fdr.csv",
        ],
        "outputs": [
            rel(OUT_DIR / "phase12_strict_atlas_discovery_lr.csv"),
            rel(OUT_DIR / "phase12_strict_atlas_discovery_topk.csv"),
            rel(OUT_DIR / "phase12_strict_atlas_discovery_dataset.csv"),
            rel(OUT_DIR / "phase12_strict_atlas_discovery_story.csv"),
            rel(OUT_DIR / "phase12_strict_atlas_discovery_status.md"),
        ],
        "claim_boundary": "Strict top-k control audit plus strict post-hoc permutation FDR; not end-to-end retraining FDR.",
    }
    (OUT_DIR / "phase12_strict_atlas_discovery_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def pick(table: pd.DataFrame, **filters: object) -> dict[str, Any]:
    if table.empty:
        return {}
    sub = table
    for key, value in filters.items():
        sub = sub[sub[key].eq(value)]
    return sub.iloc[0].to_dict() if len(sub) else {}


def get(row: dict[str, Any], key: str, default: Any = None) -> Any:
    return row[key] if key in row and not pd.isna(row[key]) else default


def best_true_lr(sub: pd.DataFrame) -> str:
    true = sub[sub["true_lr"].astype(bool)].sort_values(["strict_rank_position", "lr_pair"])
    return str(true.iloc[0]["lr_pair"]) if len(true) else ""


def best_true_lr_fdr(sub: pd.DataFrame) -> float:
    true = sub[sub["true_lr"].astype(bool)].sort_values(["strict_rank_position", "lr_pair"])
    return float(true.iloc[0]["strict_conservative_global_bh_fdr"]) if len(true) else np.nan


def fmt(value: object) -> str:
    try:
        if pd.isna(value):
            return "NA"
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(val) < 0.001 and val != 0:
        return f"{val:.2e}"
    if abs(val) >= 10:
        return f"{val:.2f}"
    return f"{val:.3g}"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
