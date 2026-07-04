#!/usr/bin/env python
"""Summarize expanded Phase 12 atlas scoring and contamination risk."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.phase12.summarize_phase12_atlas_quality as quality_mod


PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
EXPANDED_DIR = PHASE12_DIR / "atlas_model_expanded"
OUT_DIR = EXPANDED_DIR
TOP_K_VALUES = (10, 20, 50, 100)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ranking = pd.read_csv(EXPANDED_DIR / "phase12_atlas_model_expanded_lr_ranking.csv")
    hard = read_csv(EXPANDED_DIR / "phase12_atlas_model_expanded_hard_negative_leakage.csv")
    status = read_csv(EXPANDED_DIR / "phase12_atlas_model_expanded_status.csv")

    ranking = annotate_ranking(ranking)
    topk = summarize_topk(ranking)
    category = summarize_category(ranking)
    dataset = summarize_dataset(ranking, topk, hard, status)

    quality_mod.PILOT_DIR = EXPANDED_DIR
    quality_mod.OUT_DIR = EXPANDED_DIR
    quality_mod.TOP_KS = [10, 20, 50, 100]
    modules = quality_mod.build_modules(ranking)

    topk.to_csv(OUT_DIR / "phase12_expanded_atlas_topk_summary.csv", index=False)
    category.to_csv(OUT_DIR / "phase12_expanded_atlas_category_summary.csv", index=False)
    dataset.to_csv(OUT_DIR / "phase12_expanded_atlas_dataset_summary.csv", index=False)
    modules.to_csv(OUT_DIR / "phase12_expanded_atlas_modules.csv", index=False)
    write_status(dataset, topk, modules)
    write_manifest()

    print(dataset.to_string(index=False))
    print()
    print(topk.to_string(index=False))


def annotate_ranking(ranking: pd.DataFrame) -> pd.DataFrame:
    out = ranking.copy()
    out["true_lr"] = out["true_lr"].astype(bool)
    out["false_positive_high_expression"] = out["false_positive_high_expression"].astype(bool)
    out["rank"] = pd.to_numeric(out["rank"], errors="coerce")
    out["mean_score"] = pd.to_numeric(out["mean_score"], errors="coerce")
    out["n_nonzero_expr"] = pd.to_numeric(out["n_nonzero_expr"], errors="coerce").fillna(0)
    out["zero_expr_lr"] = out["n_nonzero_expr"].le(0)
    out["receptor_shared_risk"] = False
    out["ligand_shared_risk"] = False
    for dataset_key, sub in out.groupby("dataset_key"):
        true_receptors = set(sub[sub["true_lr"]]["receptor"].astype(str))
        true_ligands = set(sub[sub["true_lr"]]["ligand"].astype(str))
        mask = out["dataset_key"].eq(dataset_key)
        out.loc[mask, "receptor_shared_risk"] = (
            out.loc[mask, "receptor"].astype(str).isin(true_receptors) & ~out.loc[mask, "true_lr"].astype(bool)
        )
        out.loc[mask, "ligand_shared_risk"] = (
            out.loc[mask, "ligand"].astype(str).isin(true_ligands) & ~out.loc[mask, "true_lr"].astype(bool)
        )
    out["negative_or_risk"] = (
        out["false_positive_high_expression"]
        | out["receptor_shared_risk"]
        | out["ligand_shared_risk"]
        | out["zero_expr_lr"]
    )
    return out


def summarize_topk(ranking: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_key, sub in ranking.groupby("dataset_key"):
        sub = sub.sort_values("rank")
        for top_k in TOP_K_VALUES:
            top = sub.head(top_k)
            if top.empty:
                continue
            n_lr = len(top)
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "top_k": top_k,
                    "n_lr": n_lr,
                    "n_true_lr": int(top["true_lr"].sum()),
                    "true_lr_fraction": safe_div(top["true_lr"].sum(), n_lr),
                    "n_high_expression_negative": int(top["false_positive_high_expression"].sum()),
                    "high_expression_negative_fraction": safe_div(top["false_positive_high_expression"].sum(), n_lr),
                    "n_receptor_shared_risk": int(top["receptor_shared_risk"].sum()),
                    "receptor_shared_risk_fraction": safe_div(top["receptor_shared_risk"].sum(), n_lr),
                    "n_ligand_shared_risk": int(top["ligand_shared_risk"].sum()),
                    "ligand_shared_risk_fraction": safe_div(top["ligand_shared_risk"].sum(), n_lr),
                    "n_zero_expr_lr": int(top["zero_expr_lr"].sum()),
                    "zero_expr_lr_fraction": safe_div(top["zero_expr_lr"].sum(), n_lr),
                    "n_any_negative_or_risk": int(top["negative_or_risk"].sum()),
                    "negative_or_risk_fraction": safe_div(top["negative_or_risk"].sum(), n_lr),
                    "control_panel_empirical_fdr": safe_div(
                        top["negative_or_risk"].sum(),
                        max(int(top["negative_or_risk"].sum()) + int(top["true_lr"].sum()), 1),
                    ),
                    "top_biological_category": mode_or_empty(top["biological_category"]),
                    "top_lr_pairs": ";".join(top["lr_pair"].astype(str).head(12)),
                }
            )
    return pd.DataFrame(rows)


def summarize_category(ranking: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset_key, category), sub in ranking.groupby(["dataset_key", "biological_category"]):
        sub = sub.sort_values("rank")
        rows.append(
            {
                "dataset_key": dataset_key,
                "biological_category": category,
                "n_lr": int(len(sub)),
                "best_rank": int(sub["rank"].min()),
                "median_rank": float(sub["rank"].median()),
                "n_true_lr": int(sub["true_lr"].sum()),
                "n_high_expression_negative": int(sub["false_positive_high_expression"].sum()),
                "n_zero_expr_lr": int(sub["zero_expr_lr"].sum()),
                "n_any_negative_or_risk": int(sub["negative_or_risk"].sum()),
                "top_lr_pair": str(sub.iloc[0]["lr_pair"]),
                "top_lr_is_true": bool(sub.iloc[0]["true_lr"]),
                "top_lr_is_negative_or_risk": bool(sub.iloc[0]["negative_or_risk"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset_key", "best_rank", "biological_category"])


def summarize_dataset(
    ranking: pd.DataFrame,
    topk: pd.DataFrame,
    hard: pd.DataFrame,
    status: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for dataset_key, sub in ranking.groupby("dataset_key"):
        t10 = one_row(topk, dataset_key, 10)
        t50 = one_row(topk, dataset_key, 50)
        hard_sub = hard[hard["dataset_key"].eq(dataset_key)] if not hard.empty else pd.DataFrame()
        status_sub = status[status["dataset_key"].eq(dataset_key)].head(1) if not status.empty else pd.DataFrame()
        n_beats = int(hard_sub["negative_beats_positive"].sum()) if not hard_sub.empty else 0
        n_hard = int(len(hard_sub))
        rows.append(
            {
                "dataset_key": dataset_key,
                "n_lr": int(len(sub)),
                "n_true_lr": int(sub["true_lr"].sum()),
                "n_high_expression_negative": int(sub["false_positive_high_expression"].sum()),
                "n_zero_expr_lr": int(sub["zero_expr_lr"].sum()),
                "top10_true_lr": int(t10.get("n_true_lr", 0)),
                "top10_negative_or_risk_fraction": float(t10.get("negative_or_risk_fraction", np.nan)),
                "top10_zero_expr_fraction": float(t10.get("zero_expr_lr_fraction", np.nan)),
                "top10_control_panel_empirical_fdr": float(t10.get("control_panel_empirical_fdr", np.nan)),
                "top50_true_lr": int(t50.get("n_true_lr", 0)),
                "top50_negative_or_risk_fraction": float(t50.get("negative_or_risk_fraction", np.nan)),
                "top50_zero_expr_fraction": float(t50.get("zero_expr_lr_fraction", np.nan)),
                "hard_negative_best_comparison_beats": n_beats,
                "hard_negative_best_comparisons": n_hard,
                "hard_negative_best_comparison_beat_fraction": safe_div(n_beats, n_hard),
                "runtime_seconds": float(status_sub["runtime_seconds"].iloc[0]) if len(status_sub) else np.nan,
                "n_candidates": int(status_sub["n_candidates"].iloc[0]) if len(status_sub) else 0,
                "expanded_audit_call": classify_expanded(t10, t50, n_beats, n_hard),
            }
        )
    return pd.DataFrame(rows).sort_values("dataset_key")


def classify_expanded(t10: dict[str, object], t50: dict[str, object], n_beats: int, n_hard: int) -> str:
    top10_true = int(t10.get("n_true_lr", 0))
    top10_risk = float(t10.get("negative_or_risk_fraction", np.nan))
    top10_zero = float(t10.get("zero_expr_lr_fraction", np.nan))
    beat_fraction = safe_div(n_beats, n_hard)
    if top10_true >= 3 and top10_risk <= 0.4 and beat_fraction <= 0.2:
        return "expanded_rank_support_with_residual_risk"
    if top10_zero >= 0.3 or top10_risk >= 0.6:
        return "expanded_full_atlas_calibration_needed"
    return "expanded_mixed_or_incomplete"


def write_status(dataset: pd.DataFrame, topk: pd.DataFrame, modules: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Expanded Atlas Audit",
        "",
        "## Scope",
        "",
        "This summarizes expanded SpatialSelf atlas scoring over covered LR pairs, not the earlier 55-LR pilot.",
        "The FDR values here are still empirical risk-panel summaries, not permutation-calibrated whole-atlas FDR.",
        "",
        "## Dataset Summary",
        "",
    ]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.expanded_audit_call}`: LR={int(row.n_lr)}, "
            f"top10 true={int(row.top10_true_lr)}, top10 risk={fmt(row.top10_negative_or_risk_fraction)}, "
            f"top10 zero-expr={fmt(row.top10_zero_expr_fraction)}, hard-negative best-beat={int(row.hard_negative_best_comparison_beats)}/{int(row.hard_negative_best_comparisons)}, "
            f"candidates={int(row.n_candidates)}, runtime={float(row.runtime_seconds):.1f}s"
        )
    lines += ["", "## Top-K Contamination", ""]
    for row in topk[topk["top_k"].isin([10, 50, 100])].itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` top{int(row.top_k)}: true={int(row.n_true_lr)}, "
            f"high-neg={int(row.n_high_expression_negative)}, receptor-risk={int(row.n_receptor_shared_risk)}, "
            f"zero-expr={int(row.n_zero_expr_lr)}, risk fraction={fmt(row.negative_or_risk_fraction)}, "
            f"top category={row.top_biological_category}"
        )
    if not modules.empty:
        lines += ["", "## Communication Modules", ""]
        for row in modules.itertuples(index=False):
            warning = f"; {row.module_warning}" if str(row.module_warning) else ""
            lines.append(
                f"- `{row.dataset_key}` {row.module}: n={int(row.n_lr)}, best={int(row.best_rank)}, "
                f"anchors={int(row.n_priority_anchor)}, negatives={int(row.n_high_expression_negative)}, "
                f"category={row.top_biological_category}{warning}"
            )
    lines += [
        "",
        "## Claim Boundary",
        "",
        "- Expanded scoring now covers the large LR universe for two full-transcriptome backgrounds.",
        "- The full atlas exposes a calibration problem: top-ranked zero-expression or broad 'other' LR can dominate.",
        "- Do not claim final whole-atlas superiority until permutation-calibrated FDR and baseline whole-atlas comparison are added.",
    ]
    (OUT_DIR / "phase12_expanded_atlas_audit_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest() -> None:
    manifest = {
        "task": "phase12_expanded_atlas_audit",
        "inputs": [
            "outputs/phase12/atlas_model_expanded/phase12_atlas_model_expanded_lr_ranking.csv",
            "outputs/phase12/atlas_model_expanded/phase12_atlas_model_expanded_hard_negative_leakage.csv",
            "outputs/phase12/atlas_model_expanded/phase12_atlas_model_expanded_status.csv",
        ],
        "outputs": [
            "phase12_expanded_atlas_dataset_summary.csv",
            "phase12_expanded_atlas_topk_summary.csv",
            "phase12_expanded_atlas_category_summary.csv",
            "phase12_expanded_atlas_modules.csv",
            "phase12_expanded_atlas_audit_status.md",
        ],
        "claim_boundary": "Expanded covered-LR scoring audit; not final permutation-calibrated FDR or baseline superiority.",
    }
    (OUT_DIR / "phase12_expanded_atlas_audit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def one_row(topk: pd.DataFrame, dataset_key: str, top_k: int) -> dict[str, object]:
    row = topk[(topk["dataset_key"].eq(dataset_key)) & (topk["top_k"].eq(top_k))].head(1)
    return row.iloc[0].to_dict() if len(row) else {}


def mode_or_empty(values: pd.Series) -> str:
    if values.empty:
        return ""
    counts = values.astype(str).value_counts()
    return str(counts.index[0]) if len(counts) else ""


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else np.nan


def fmt(value: object) -> str:
    try:
        if value is None or pd.isna(value):
            return "NA"
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{val:.3g}"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


if __name__ == "__main__":
    main()
