#!/usr/bin/env python
"""Atlas-level receptor-shared negative audit for Phase 12.

This complements the hard-negative stress table by asking the same question on
the full expanded LR ranking: when a non-anchor LR uses the same receptor as a
pre-specified positive anchor, does it outrank that anchor?
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
OUT_DIR = PHASE12_DIR / "receptor_shared_atlas_audit"
TOP_K_VALUES = (10, 20, 50, 100, 200)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ranking = pd.read_csv(EXPANDED_DIR / "phase12_atlas_model_expanded_calibrated_lr_ranking.csv")
    annotated = annotate_receptor_shared(ranking)
    pair_audit = build_pair_audit(annotated)
    topk = summarize_topk(annotated)
    dataset = summarize_dataset(annotated, pair_audit, topk)

    annotated.to_csv(OUT_DIR / "phase12_receptor_shared_atlas_decoy_table.csv", index=False)
    pair_audit.to_csv(OUT_DIR / "phase12_receptor_shared_atlas_positive_pair_audit.csv", index=False)
    topk.to_csv(OUT_DIR / "phase12_receptor_shared_atlas_topk_summary.csv", index=False)
    dataset.to_csv(OUT_DIR / "phase12_receptor_shared_atlas_dataset_summary.csv", index=False)
    write_status(dataset, pair_audit, topk)
    write_manifest(annotated, pair_audit, topk, dataset)
    print(dataset.to_string(index=False))
    print()
    print(pair_audit.sort_values(["dataset_key", "positive_rank"]).to_string(index=False))


def annotate_receptor_shared(ranking: pd.DataFrame) -> pd.DataFrame:
    out = ranking.copy()
    out["rank"] = pd.to_numeric(out["rank"], errors="coerce")
    out["rank_q95"] = pd.to_numeric(out.get("rank_q95", np.nan), errors="coerce")
    out["calibrated_score"] = pd.to_numeric(out.get("calibrated_score", np.nan), errors="coerce")
    out["true_lr"] = out["true_lr"].fillna(False).astype(bool)
    out["false_positive_high_expression"] = out["false_positive_high_expression"].fillna(False).astype(bool)
    out["receptor_shared_decoy"] = False
    out["shared_receptor_positive_lr"] = ""
    out["n_positive_anchors_for_receptor"] = 0
    out["receptor_shared_decoy_class"] = "not_receptor_shared"
    for dataset_key, sub in out.groupby("dataset_key", sort=False):
        positives = sub[sub["true_lr"]].copy()
        receptor_to_pairs = positives.groupby("receptor")["lr_pair"].apply(lambda x: sorted(x.astype(str))).to_dict()
        receptor_counts = {str(k): len(v) for k, v in receptor_to_pairs.items()}
        mask = out["dataset_key"].eq(dataset_key)
        receptors = out.loc[mask, "receptor"].astype(str)
        out.loc[mask, "n_positive_anchors_for_receptor"] = receptors.map(receptor_counts).fillna(0).astype(int)
        out.loc[mask, "shared_receptor_positive_lr"] = receptors.map(
            {str(k): ";".join(v) for k, v in receptor_to_pairs.items()}
        ).fillna("")
        out.loc[mask, "receptor_shared_decoy"] = (
            out.loc[mask, "n_positive_anchors_for_receptor"].gt(0)
            & ~out.loc[mask, "true_lr"].astype(bool)
        )
    out.loc[out["receptor_shared_decoy"], "receptor_shared_decoy_class"] = "receptor_shared_background"
    out.loc[
        out["receptor_shared_decoy"] & out["false_positive_high_expression"],
        "receptor_shared_decoy_class",
    ] = "receptor_shared_high_expression"
    out.loc[out["true_lr"], "receptor_shared_decoy_class"] = "positive_anchor"
    return out.sort_values(["dataset_key", "rank", "lr_pair"]).reset_index(drop=True)


def build_pair_audit(annotated: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_key, sub in annotated.groupby("dataset_key", sort=False):
        positives = sub[sub["true_lr"]].copy()
        for pos in positives.itertuples(index=False):
            same_receptor = sub[
                sub["receptor"].astype(str).eq(str(pos.receptor))
                & ~sub["true_lr"].astype(bool)
            ].copy()
            same_receptor = same_receptor.sort_values(["rank", "lr_pair"])
            best = same_receptor.head(1)
            n_decoy = int(len(same_receptor))
            n_decoy_beats = int(pd.to_numeric(same_receptor["rank"], errors="coerce").lt(float(pos.rank)).sum())
            n_high_expr_beats = int(
                (
                    same_receptor["false_positive_high_expression"].astype(bool)
                    & pd.to_numeric(same_receptor["rank"], errors="coerce").lt(float(pos.rank))
                ).sum()
            )
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "positive_lr_id": pos.lr_id,
                    "positive_lr_pair": pos.lr_pair,
                    "positive_ligand": pos.ligand,
                    "shared_receptor": pos.receptor,
                    "positive_rank": int(pos.rank),
                    "positive_rank_q95": safe_float(pos.rank_q95),
                    "positive_score": safe_float(pos.calibrated_score),
                    "n_receptor_shared_decoys": n_decoy,
                    "n_receptor_shared_decoys_beating_positive": n_decoy_beats,
                    "n_high_expression_receptor_shared_decoys_beating_positive": n_high_expr_beats,
                    "best_decoy_lr_pair": str(best["lr_pair"].iloc[0]) if len(best) else "",
                    "best_decoy_rank": int(best["rank"].iloc[0]) if len(best) else -1,
                    "best_decoy_rank_q95": safe_float(best["rank_q95"].iloc[0]) if len(best) else np.nan,
                    "best_decoy_score": safe_float(best["calibrated_score"].iloc[0]) if len(best) else np.nan,
                    "best_decoy_high_expression": bool(best["false_positive_high_expression"].iloc[0]) if len(best) else False,
                    "best_decoy_selected_reason": str(best["selected_reason"].iloc[0]) if len(best) else "",
                    "best_decoy_bio_category": str(best["biological_category"].iloc[0]) if len(best) else "",
                    "rank_margin_positive_minus_best_decoy": (
                        int(pos.rank) - int(best["rank"].iloc[0]) if len(best) else np.nan
                    ),
                    "receptor_shared_control_call": pair_call(n_decoy, n_decoy_beats, n_high_expr_beats),
                }
            )
    return pd.DataFrame(rows)


def summarize_topk(annotated: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_key, sub in annotated.groupby("dataset_key", sort=False):
        sub = sub.sort_values(["rank", "lr_pair"]).copy()
        for top_k in TOP_K_VALUES:
            top = sub.head(top_k)
            receptor_decoys = top[top["receptor_shared_decoy"].astype(bool)]
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "top_k": top_k,
                    "n_lr": int(len(top)),
                    "n_true_lr": int(top["true_lr"].sum()),
                    "n_receptor_shared_decoy": int(len(receptor_decoys)),
                    "n_receptor_shared_high_expression_decoy": int(
                        receptor_decoys["false_positive_high_expression"].astype(bool).sum()
                    ),
                    "receptor_shared_contamination_fraction": float(len(receptor_decoys) / max(len(top), 1)),
                    "receptor_shared_decoy_lr_pairs": ";".join(receptor_decoys["lr_pair"].astype(str).head(12)),
                }
            )
    return pd.DataFrame(rows)


def summarize_dataset(annotated: pd.DataFrame, pair_audit: pd.DataFrame, topk: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_key, sub in annotated.groupby("dataset_key", sort=False):
        pairs = pair_audit[pair_audit["dataset_key"].eq(dataset_key)]
        top = topk[topk["dataset_key"].eq(dataset_key)]
        top10 = top[top["top_k"].eq(10)].head(1)
        top50 = top[top["top_k"].eq(50)].head(1)
        top100 = top[top["top_k"].eq(100)].head(1)
        n_pos_with_decoys = int(pairs["n_receptor_shared_decoys"].gt(0).sum()) if not pairs.empty else 0
        n_pos_decoy_beats = int(pairs["n_receptor_shared_decoys_beating_positive"].gt(0).sum()) if not pairs.empty else 0
        n_pos_high_expr_beats = int(
            pairs["n_high_expression_receptor_shared_decoys_beating_positive"].gt(0).sum()
        ) if not pairs.empty else 0
        total_pairwise_beats = int(pairs["n_receptor_shared_decoys_beating_positive"].sum()) if not pairs.empty else 0
        rows.append(
            {
                "dataset_key": dataset_key,
                "n_lr": int(len(sub)),
                "n_positive_anchor": int(sub["true_lr"].sum()),
                "n_receptor_shared_decoy": int(sub["receptor_shared_decoy"].sum()),
                "n_receptor_shared_high_expression_decoy": int(
                    (sub["receptor_shared_decoy"].astype(bool) & sub["false_positive_high_expression"].astype(bool)).sum()
                ),
                "n_positive_anchors_with_receptor_shared_decoys": n_pos_with_decoys,
                "n_positive_anchors_beaten_by_receptor_shared_decoy": n_pos_decoy_beats,
                "n_positive_anchors_beaten_by_high_expression_receptor_shared_decoy": n_pos_high_expr_beats,
                "total_receptor_shared_decoys_beating_positive": total_pairwise_beats,
                "top10_receptor_shared_decoy": value(top10, "n_receptor_shared_decoy"),
                "top50_receptor_shared_decoy": value(top50, "n_receptor_shared_decoy"),
                "top100_receptor_shared_decoy": value(top100, "n_receptor_shared_decoy"),
                "top10_receptor_shared_contamination_fraction": value(top10, "receptor_shared_contamination_fraction"),
                "top50_receptor_shared_contamination_fraction": value(top50, "receptor_shared_contamination_fraction"),
                "top100_receptor_shared_contamination_fraction": value(top100, "receptor_shared_contamination_fraction"),
                "best_receptor_shared_decoy_rank": best_rank(sub),
                "best_receptor_shared_decoy_lr_pair": best_pair(sub),
                "receptor_shared_atlas_call": dataset_call(n_pos_decoy_beats, total_pairwise_beats, value(top10, "n_receptor_shared_decoy")),
            }
        )
    return pd.DataFrame(rows)


def pair_call(n_decoy: int, n_decoy_beats: int, n_high_expr_beats: int) -> str:
    if n_decoy == 0:
        return "no_same_receptor_decoys_available"
    if n_decoy_beats == 0:
        return "positive_anchor_beats_all_same_receptor_decoys"
    if n_high_expr_beats > 0:
        return "high_expression_same_receptor_decoy_beats_positive"
    return "same_receptor_decoy_beats_positive"


def dataset_call(n_pos_decoy_beats: int, total_pairwise_beats: int, top10_decoys: float) -> str:
    if n_pos_decoy_beats == 0 and top10_decoys == 0:
        return "receptor_shared_controls_suppressed_in_top10_and_pairwise"
    if top10_decoys == 0 and n_pos_decoy_beats <= 2:
        return "top10_clean_low_pairwise_receptor_shared_leakage"
    if top10_decoys <= 1:
        return "top10_low_receptor_shared_leakage_pairwise_caveat"
    return "receptor_shared_leakage_needs_calibration"


def best_rank(sub: pd.DataFrame) -> int:
    decoy = sub[sub["receptor_shared_decoy"].astype(bool)].sort_values(["rank", "lr_pair"]).head(1)
    return int(decoy["rank"].iloc[0]) if len(decoy) else -1


def best_pair(sub: pd.DataFrame) -> str:
    decoy = sub[sub["receptor_shared_decoy"].astype(bool)].sort_values(["rank", "lr_pair"]).head(1)
    return str(decoy["lr_pair"].iloc[0]) if len(decoy) else ""


def value(table: pd.DataFrame, column: str) -> float:
    if table.empty or column not in table:
        return np.nan
    return float(table[column].iloc[0])


def safe_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def write_status(dataset: pd.DataFrame, pair_audit: pd.DataFrame, topk: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Receptor-Shared Atlas Audit",
        "",
        "## Scope",
        "",
        "This audit tests receptor-shared hard negatives on the full expanded LR ranking. It is a control-contamination audit, not native ground-truth FDR for real tissue.",
        "",
        "## Dataset Summary",
        "",
    ]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.receptor_shared_atlas_call}`: "
            f"receptor-shared decoys={int(row.n_receptor_shared_decoy)}, "
            f"positive anchors beaten={int(row.n_positive_anchors_beaten_by_receptor_shared_decoy)}/{int(row.n_positive_anchor)}, "
            f"pairwise beats={int(row.total_receptor_shared_decoys_beating_positive)}, "
            f"top10 decoys={int(row.top10_receptor_shared_decoy)}, top50 decoys={int(row.top50_receptor_shared_decoy)}, "
            f"best decoy={row.best_receptor_shared_decoy_lr_pair}#{int(row.best_receptor_shared_decoy_rank)}"
        )
    lines += ["", "## Positive Anchor Pair Audit", ""]
    for row in pair_audit.sort_values(["dataset_key", "positive_rank"]).itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.positive_lr_pair}` rank {int(row.positive_rank)} receptor `{row.shared_receptor}`: "
            f"decoys={int(row.n_receptor_shared_decoys)}, beats={int(row.n_receptor_shared_decoys_beating_positive)}, "
            f"best decoy={row.best_decoy_lr_pair}#{int(row.best_decoy_rank)}; call={row.receptor_shared_control_call}"
        )
    lines += [
        "",
        "## Claim Boundary",
        "",
        "Use this as atlas-level evidence for receptor-shared false-positive contamination or suppression. It does not replace the conservative whole-atlas permutation FDR audit.",
    ]
    (OUT_DIR / "phase12_receptor_shared_atlas_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(annotated: pd.DataFrame, pair_audit: pd.DataFrame, topk: pd.DataFrame, dataset: pd.DataFrame) -> None:
    manifest = {
        "task": "phase12_receptor_shared_atlas_audit",
        "outputs": [
            rel(OUT_DIR / "phase12_receptor_shared_atlas_decoy_table.csv"),
            rel(OUT_DIR / "phase12_receptor_shared_atlas_positive_pair_audit.csv"),
            rel(OUT_DIR / "phase12_receptor_shared_atlas_topk_summary.csv"),
            rel(OUT_DIR / "phase12_receptor_shared_atlas_dataset_summary.csv"),
            rel(OUT_DIR / "phase12_receptor_shared_atlas_status.md"),
        ],
        "n_lr_rows": int(len(annotated)),
        "n_pair_rows": int(len(pair_audit)),
        "n_topk_rows": int(len(topk)),
        "n_dataset_rows": int(len(dataset)),
        "claim_boundary": "Atlas-level receptor-shared control audit, not native whole-atlas discovery FDR.",
    }
    (OUT_DIR / "phase12_receptor_shared_atlas_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
