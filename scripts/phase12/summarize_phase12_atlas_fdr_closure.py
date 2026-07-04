#!/usr/bin/env python
"""Closure audit for Phase 12 expanded whole-atlas FDR evidence.

This script does not try to turn the current atlas FDR into a positive
discovery result. It separates three ideas that are easy to conflate:

1. strict post-hoc permutation discovery FDR;
2. hard-control/target-decoy contamination FDR;
3. top-k biological prioritization with control caveats.

The resulting table is meant to make the remaining Goal 1 blocker explicit.
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
DECOY_DIR = EXPANDED_DIR / "target_decoy_fdr"
OUT_DIR = PHASE12_DIR / "atlas_fdr_closure"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ranking = read_csv(EXPANDED_DIR / "phase12_atlas_model_expanded_calibrated_lr_ranking.csv")
    perm_dataset = read_csv(PERM_DIR / "phase12_expanded_atlas_permutation_dataset_summary.csv")
    perm_lr = read_csv(PERM_DIR / "phase12_expanded_atlas_permutation_lr_fdr.csv")
    decoy_dataset = read_csv(DECOY_DIR / "phase12_expanded_target_decoy_dataset_summary.csv")
    decoy_lr = read_csv(DECOY_DIR / "phase12_expanded_target_decoy_lr_fdr.csv")
    receptor = read_csv(PHASE12_DIR / "receptor_shared_atlas_audit/phase12_receptor_shared_atlas_dataset_summary.csv")
    if ranking.empty:
        raise FileNotFoundError("Missing calibrated expanded atlas ranking")

    dataset = build_dataset_closure(ranking, perm_dataset, decoy_dataset, receptor)
    top_lr = build_top_lr_closure(ranking, perm_lr, decoy_lr)
    story = build_story_closure(dataset)

    dataset.to_csv(OUT_DIR / "phase12_atlas_fdr_closure_dataset.csv", index=False)
    top_lr.to_csv(OUT_DIR / "phase12_atlas_fdr_closure_top_lr.csv", index=False)
    story.to_csv(OUT_DIR / "phase12_atlas_fdr_closure_story.csv", index=False)
    write_status(dataset, top_lr, story)
    write_manifest(dataset, top_lr, story)
    print(dataset.to_string(index=False))
    print()
    print(top_lr.groupby("dataset_key").head(12).to_string(index=False))


def build_dataset_closure(
    ranking: pd.DataFrame,
    perm_dataset: pd.DataFrame,
    decoy_dataset: pd.DataFrame,
    receptor: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_key, sub in ranking.groupby("dataset_key", sort=False):
        sub = sub.sort_values("rank")
        perm = pick(perm_dataset, dataset_key=dataset_key)
        decoy = pick(decoy_dataset, dataset_key=dataset_key)
        rec = pick(receptor, dataset_key=dataset_key)
        top10 = sub.head(10)
        top100 = sub.head(100)
        n_perm_hits = int(get(perm, "all_lr_conservative_fdr_0_1", 0) or 0)
        top10_control_fdr = safe_float(get(decoy, "top10_primary_control_prefix_fdr", np.nan))
        top100_control_fdr = safe_float(get(decoy, "top100_primary_control_prefix_fdr", np.nan))
        top10_receptor_decoy = int(get(rec, "top10_receptor_shared_decoy", 0) or 0)
        rows.append(
            {
                "dataset_key": dataset_key,
                "n_lr": int(len(sub)),
                "n_true_lr": int(pd.to_numeric(sub.get("true_lr", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
                "top10_true_lr": int(top10["true_lr"].fillna(False).astype(bool).sum()),
                "top10_high_expression_negative": int(top10["false_positive_high_expression"].fillna(False).astype(bool).sum()),
                "top10_primary_control_fdr": top10_control_fdr,
                "top100_primary_control_fdr": top100_control_fdr,
                "top10_receptor_shared_decoy": top10_receptor_decoy,
                "all_lr_permutation_q_0_1": n_perm_hits,
                "best_true_lr": get(perm, "best_true_lr", ""),
                "best_true_lr_permutation_fdr": safe_float(get(perm, "best_true_lr_fdr", np.nan)),
                "top10_unlabeled_background": int(
                    top10.get("selected_reason", pd.Series("", index=top10.index)).astype(str).eq("category_balanced").sum()
                ),
                "top100_unlabeled_background": int(
                    top100.get("selected_reason", pd.Series("", index=top100.index)).astype(str).eq("category_balanced").sum()
                ),
                "top10_ligand_shared_decoy": int(top10_ligand_shared(decoy_lr=None, dataset_key=dataset_key)),
                "dataset_fdr_closure_call": dataset_call(
                    n_perm_hits=n_perm_hits,
                    top10_control_fdr=top10_control_fdr,
                    top100_control_fdr=top100_control_fdr,
                    top10_true=int(top10["true_lr"].fillna(False).astype(bool).sum()),
                    top10_bg=int(top10.get("selected_reason", pd.Series("", index=top10.index)).astype(str).eq("category_balanced").sum()),
                ),
            }
        )
    return pd.DataFrame(rows)


def top10_ligand_shared(decoy_lr: pd.DataFrame | None, dataset_key: str) -> int:
    # Kept for schema stability; top-LR table carries exact decoy annotations.
    return 0


def build_top_lr_closure(ranking: pd.DataFrame, perm_lr: pd.DataFrame, decoy_lr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    perm_cols = [
        "dataset_key",
        "lr_id",
        "conservative_global_bh_fdr",
        "conservative_global_empirical_p",
        "receiver_permutation_global_bh_fdr",
        "spatial_distance_permutation_global_bh_fdr",
        "lr_identity_permutation_global_bh_fdr",
    ]
    decoy_cols = [
        "dataset_key",
        "lr_id",
        "decoy_class",
        "primary_control_decoy",
        "strict_control_decoy",
        "receptor_shared_decoy",
        "ligand_shared_decoy",
        "zero_expression_decoy",
        "primary_control_prefix_fdr",
        "strict_control_prefix_fdr",
    ]
    table = ranking.copy()
    if not perm_lr.empty:
        table = table.merge(perm_lr[[c for c in perm_cols if c in perm_lr.columns]], on=["dataset_key", "lr_id"], how="left")
    if not decoy_lr.empty:
        table = table.merge(decoy_lr[[c for c in decoy_cols if c in decoy_lr.columns]], on=["dataset_key", "lr_id"], how="left")
    for dataset_key, sub in table.groupby("dataset_key", sort=False):
        for row in sub.sort_values("rank").head(25).itertuples(index=False):
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "rank": int(row.rank),
                    "lr_pair": row.lr_pair,
                    "true_lr": bool(row.true_lr),
                    "false_positive_high_expression": bool(row.false_positive_high_expression),
                    "selected_reason": row.selected_reason,
                    "biological_category": row.biological_category,
                    "calibrated_score": row.calibrated_score,
                    "conservative_global_bh_fdr": get(row, "conservative_global_bh_fdr", np.nan),
                    "primary_control_prefix_fdr": get(row, "primary_control_prefix_fdr", np.nan),
                    "decoy_class": get(row, "decoy_class", ""),
                    "primary_control_decoy": bool(get(row, "primary_control_decoy", False)),
                    "strict_control_decoy": bool(get(row, "strict_control_decoy", False)),
                    "receptor_shared_decoy": bool(get(row, "receptor_shared_decoy", False)),
                    "ligand_shared_decoy": bool(get(row, "ligand_shared_decoy", False)),
                    "zero_expression_decoy": bool(get(row, "zero_expression_decoy", False)),
                    "top_lr_fdr_call": top_lr_call(row),
                }
            )
    return pd.DataFrame(rows)


def top_lr_call(row: Any) -> str:
    q = safe_float(get(row, "conservative_global_bh_fdr", np.nan))
    if bool(get(row, "true_lr", False)) and np.isfinite(q) and q <= 0.1:
        return "permutation_fdr_supported_true_lr"
    if bool(get(row, "true_lr", False)):
        return "biological_anchor_ranked_but_permutation_fdr_not_supported"
    if bool(get(row, "primary_control_decoy", False)):
        return "hard_control_decoy_in_top_rank"
    if str(get(row, "selected_reason", "")) == "category_balanced":
        return "unlabeled_background_top_rank"
    return "top_rank_nonanchor"


def build_story_closure(dataset: pd.DataFrame) -> pd.DataFrame:
    n_datasets = int(len(dataset))
    n_perm_supported = int(pd.to_numeric(dataset["all_lr_permutation_q_0_1"], errors="coerce").fillna(0).sum())
    n_top10_control_clean = int(pd.to_numeric(dataset["top10_primary_control_fdr"], errors="coerce").le(0.1).sum())
    n_top100_control_clean = int(pd.to_numeric(dataset["top100_primary_control_fdr"], errors="coerce").le(0.1).sum())
    if n_perm_supported > 0:
        call = "global_permutation_discovery_supported"
    elif n_top100_control_clean >= n_datasets:
        call = "control_fdr_supported_permutation_discovery_not_supported"
    else:
        call = "global_fdr_audit_complete_discovery_blocked"
    return pd.DataFrame(
        [
            {
                "n_datasets": n_datasets,
                "n_datasets_top10_control_fdr_le_0_1": n_top10_control_clean,
                "n_datasets_top100_control_fdr_le_0_1": n_top100_control_clean,
                "n_all_lr_permutation_q_0_1": n_perm_supported,
                "overall_atlas_fdr_closure_call": call,
                "claim_boundary": "Whole-atlas FDR outputs are complete, but strict permutation FDR does not support all-LR discoveries. Use target-decoy/control FDR as contamination audit, not native discovery FDR.",
            }
        ]
    )


def dataset_call(
    *,
    n_perm_hits: int,
    top10_control_fdr: float,
    top100_control_fdr: float,
    top10_true: int,
    top10_bg: int,
) -> str:
    if n_perm_hits > 0:
        return "permutation_discovery_supported"
    if np.isfinite(top10_control_fdr) and top10_control_fdr <= 0.1 and top10_true > 0:
        return "top10_control_fdr_clean_permutation_blocked"
    if np.isfinite(top100_control_fdr) and top100_control_fdr <= 0.1 and top10_true > 0:
        return "top100_control_fdr_clean_top10_caveat_permutation_blocked"
    if top10_bg > 0:
        return "background_top_rank_contamination_permutation_blocked"
    return "fdr_audit_complete_not_supportive"


def write_status(dataset: pd.DataFrame, top_lr: pd.DataFrame, story: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Atlas FDR Closure",
        "",
        "## Scope",
        "",
        "This audit separates strict permutation FDR from target-decoy/control FDR. It is intentionally conservative: current strict all-LR permutation FDR does not support discovery claims.",
        "",
        "## Dataset Summary",
        "",
    ]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.dataset_fdr_closure_call}`: LR={int(row.n_lr)}, "
            f"top10 true={int(row.top10_true_lr)}, top10 highNeg={int(row.top10_high_expression_negative)}, "
            f"top10 control FDR={fmt(row.top10_primary_control_fdr)}, top100 control FDR={fmt(row.top100_primary_control_fdr)}, "
            f"all-LR permutation q<=0.1={int(row.all_lr_permutation_q_0_1)}, best true={row.best_true_lr} q={fmt(row.best_true_lr_permutation_fdr)}, "
            f"top10 background={int(row.top10_unlabeled_background)}."
        )
    lines += ["", "## Top-Rank Failure Modes", ""]
    for row in top_lr[top_lr["top_lr_fdr_call"].astype(str).ne("biological_anchor_ranked_but_permutation_fdr_not_supported")].head(20).itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` rank {int(row.rank)} `{row.lr_pair}`: {row.top_lr_fdr_call}; decoy={row.decoy_class}; prefixFDR={fmt(row.primary_control_prefix_fdr)}."
        )
    lines += ["", "## Overall", ""]
    if not story.empty:
        s = story.iloc[0]
        lines.append(
            f"- `{s.overall_atlas_fdr_closure_call}`: datasets={int(s.n_datasets)}, "
            f"top10-control-clean={int(s.n_datasets_top10_control_fdr_le_0_1)}, "
            f"top100-control-clean={int(s.n_datasets_top100_control_fdr_le_0_1)}, "
            f"all-LR permutation discoveries={int(s.n_all_lr_permutation_q_0_1)}."
        )
        lines.append(f"- Claim boundary: {s.claim_boundary}")
    (OUT_DIR / "phase12_atlas_fdr_closure_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(dataset: pd.DataFrame, top_lr: pd.DataFrame, story: pd.DataFrame) -> None:
    manifest = {
        "task": "phase12_atlas_fdr_closure",
        "outputs": [
            rel(OUT_DIR / "phase12_atlas_fdr_closure_dataset.csv"),
            rel(OUT_DIR / "phase12_atlas_fdr_closure_top_lr.csv"),
            rel(OUT_DIR / "phase12_atlas_fdr_closure_story.csv"),
            rel(OUT_DIR / "phase12_atlas_fdr_closure_status.md"),
        ],
        "n_dataset_rows": int(len(dataset)),
        "n_top_lr_rows": int(len(top_lr)),
        "overall_call": str(story["overall_atlas_fdr_closure_call"].iloc[0]) if not story.empty else "",
        "claim_boundary": "Global FDR audit closure, not a positive whole-atlas discovery claim.",
    }
    (OUT_DIR / "phase12_atlas_fdr_closure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def pick(table: pd.DataFrame, **criteria: Any) -> pd.Series:
    if table.empty:
        return pd.Series(dtype=object)
    mask = pd.Series(True, index=table.index)
    for key, value in criteria.items():
        if key not in table.columns:
            return pd.Series(dtype=object)
        mask &= table[key].astype(str).eq(str(value))
    if not mask.any():
        return pd.Series(dtype=object)
    return table.loc[mask].iloc[0]


def get(obj: Any, key: str, default: Any = "") -> Any:
    if isinstance(obj, pd.Series):
        return obj[key] if key in obj.index else default
    return getattr(obj, key, default)


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def fmt(value: Any) -> str:
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.3g}"
    except (TypeError, ValueError):
        return str(value)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
