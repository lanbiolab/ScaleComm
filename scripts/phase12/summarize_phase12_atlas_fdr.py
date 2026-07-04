#!/usr/bin/env python
"""Summarize Phase 12 atlas pilot ranking, empirical FDR and contamination.

This is a conservative audit layer for the pilot atlas model. It combines LR
ranking, top-k contamination, hard-negative stress and module summaries. The
empirical FDR here is control-panel based, not a final permutation-calibrated
whole-atlas FDR.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
MODEL_DIR = PHASE12_DIR / "atlas_model_pilot"
STRESS_DIR = PHASE12_DIR / "hard_negative_stress"
OUT_DIR = PHASE12_DIR / "atlas_fdr_audit"

TOP_K_VALUES = (5, 10, 20, 50)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ranking = read_csv(MODEL_DIR / "phase12_atlas_model_pilot_lr_ranking.csv")
    hard = read_csv(STRESS_DIR / "phase12_hard_negative_stress_positive_summary.csv")
    dataset_stress = read_csv(STRESS_DIR / "phase12_hard_negative_stress_dataset_summary.csv")
    modules = read_csv(MODEL_DIR / "phase12_atlas_model_pilot_modules.csv")
    if ranking.empty:
        raise FileNotFoundError("Missing atlas pilot ranking table.")

    ranking = annotate_ranking(ranking)
    topk = summarize_topk(ranking)
    category = summarize_category(ranking)
    positive = summarize_positive_controls(ranking, hard)
    dataset = summarize_dataset(ranking, topk, category, positive, dataset_stress, modules)

    dataset.to_csv(OUT_DIR / "phase12_atlas_fdr_dataset_summary.csv", index=False)
    topk.to_csv(OUT_DIR / "phase12_atlas_fdr_topk_summary.csv", index=False)
    category.to_csv(OUT_DIR / "phase12_atlas_fdr_category_summary.csv", index=False)
    positive.to_csv(OUT_DIR / "phase12_atlas_fdr_positive_control_summary.csv", index=False)
    write_status(dataset, topk, category, positive)
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
    out["receptor"] = out["receptor"].astype(str)
    out["ligand"] = out["ligand"].astype(str)
    out["biological_category"] = out["biological_category"].fillna("unknown").astype(str)
    out["mechanism"] = out["mechanism"].fillna("unknown").astype(str)
    out["receptor_shared_risk"] = False
    out["ligand_shared_risk"] = False
    for dataset_key, sub in out.groupby("dataset_key"):
        true_receptors = set(sub[sub["true_lr"]]["receptor"].astype(str))
        true_ligands = set(sub[sub["true_lr"]]["ligand"].astype(str))
        mask = out["dataset_key"].eq(dataset_key)
        out.loc[mask, "receptor_shared_risk"] = (
            out.loc[mask, "receptor"].astype(str).isin(true_receptors)
            & ~out.loc[mask, "true_lr"].astype(bool)
        )
        out.loc[mask, "ligand_shared_risk"] = (
            out.loc[mask, "ligand"].astype(str).isin(true_ligands)
            & ~out.loc[mask, "true_lr"].astype(bool)
        )
    out["negative_or_risk"] = (
        out["false_positive_high_expression"]
        | out["receptor_shared_risk"]
        | out["ligand_shared_risk"]
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
            n_true = int(top["true_lr"].sum())
            n_high = int(top["false_positive_high_expression"].sum())
            n_receptor = int(top["receptor_shared_risk"].sum())
            n_ligand = int(top["ligand_shared_risk"].sum())
            n_any_risk = int(top["negative_or_risk"].sum())
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "top_k": top_k,
                    "n_lr": n_lr,
                    "n_true_lr": n_true,
                    "true_lr_fraction": safe_div(n_true, n_lr),
                    "n_high_expression_negative": n_high,
                    "high_expression_negative_fraction": safe_div(n_high, n_lr),
                    "n_receptor_shared_risk": n_receptor,
                    "receptor_shared_risk_fraction": safe_div(n_receptor, n_lr),
                    "n_ligand_shared_risk": n_ligand,
                    "ligand_shared_risk_fraction": safe_div(n_ligand, n_lr),
                    "n_any_negative_or_risk": n_any_risk,
                    "negative_or_risk_fraction": safe_div(n_any_risk, n_lr),
                    "control_panel_empirical_fdr": safe_div(n_any_risk, max(n_any_risk + n_true, 1)),
                    "top_biological_category": mode_or_empty(top["biological_category"]),
                    "top_mechanism": mode_or_empty(top["mechanism"]),
                    "top_lr_pairs": ";".join(top["lr_pair"].astype(str).head(10)),
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
                "best_rank": int(sub["rank"].min()) if len(sub) else -1,
                "median_rank": float(sub["rank"].median()) if len(sub) else np.nan,
                "n_true_lr": int(sub["true_lr"].sum()),
                "n_high_expression_negative": int(sub["false_positive_high_expression"].sum()),
                "n_receptor_shared_risk": int(sub["receptor_shared_risk"].sum()),
                "n_any_negative_or_risk": int(sub["negative_or_risk"].sum()),
                "top_lr_pair": str(sub.iloc[0]["lr_pair"]) if len(sub) else "",
                "top_lr_is_true": bool(sub.iloc[0]["true_lr"]) if len(sub) else False,
                "top_lr_is_negative_or_risk": bool(sub.iloc[0]["negative_or_risk"]) if len(sub) else False,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["dataset_key", "best_rank", "biological_category"])


def summarize_positive_controls(ranking: pd.DataFrame, hard: pd.DataFrame) -> pd.DataFrame:
    rows = []
    hard = hard.copy() if not hard.empty else pd.DataFrame()
    for dataset_key, sub in ranking.groupby("dataset_key"):
        positives = sub[sub["true_lr"].astype(bool)].sort_values("rank")
        hard_sub = hard[hard["dataset_key"].eq(dataset_key)] if not hard.empty else pd.DataFrame()
        for row in positives.itertuples(index=False):
            stress = hard_sub[hard_sub["positive_lr_pair"].astype(str).eq(str(row.lr_pair))].head(1)
            out = {
                "dataset_key": dataset_key,
                "lr_pair": row.lr_pair,
                "rank": int(row.rank),
                "score": float(row.mean_score),
                "biological_category": row.biological_category,
                "mechanism": row.mechanism,
                "top10": bool(row.rank <= 10),
                "top20": bool(row.rank <= 20),
                "negative_beat_fraction": np.nan,
                "empirical_p_score": np.nan,
                "empirical_bh_fdr_score": np.nan,
                "stress_pass_rank_score": False,
                "stress_pass_downstream": False,
                "best_negative_pair": "",
                "best_negative_type": "",
                "best_negative_rank": np.nan,
                "best_negative_minus_positive_score": np.nan,
            }
            if len(stress):
                s = stress.iloc[0]
                out.update(
                    {
                        "negative_beat_fraction": float(s.get("negative_beat_fraction", np.nan)),
                        "empirical_p_score": float(s.get("empirical_p_score", np.nan)),
                        "empirical_bh_fdr_score": float(s.get("empirical_bh_fdr_score", np.nan)),
                        "stress_pass_rank_score": bool(s.get("stress_pass_rank_score", False)),
                        "stress_pass_downstream": bool(s.get("stress_pass_downstream", False)),
                        "best_negative_pair": str(s.get("best_negative_pair", "")),
                        "best_negative_type": str(s.get("best_negative_type", "")),
                        "best_negative_rank": float(s.get("best_negative_rank", np.nan)),
                        "best_negative_minus_positive_score": float(s.get("best_negative_minus_positive_score", np.nan)),
                    }
                )
            rows.append(out)
    return pd.DataFrame(rows)


def summarize_dataset(
    ranking: pd.DataFrame,
    topk: pd.DataFrame,
    category: pd.DataFrame,
    positive: pd.DataFrame,
    dataset_stress: pd.DataFrame,
    modules: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for dataset_key, sub in ranking.groupby("dataset_key"):
        top10 = topk[(topk["dataset_key"].eq(dataset_key)) & (topk["top_k"].eq(10))].head(1)
        top20 = topk[(topk["dataset_key"].eq(dataset_key)) & (topk["top_k"].eq(20))].head(1)
        pos = positive[positive["dataset_key"].eq(dataset_key)] if not positive.empty else pd.DataFrame()
        stress = dataset_stress[dataset_stress["dataset_key"].eq(dataset_key)].head(1) if not dataset_stress.empty else pd.DataFrame()
        mod = modules[modules["dataset_key"].eq(dataset_key)] if not modules.empty else pd.DataFrame()
        cat = category[category["dataset_key"].eq(dataset_key)] if not category.empty else pd.DataFrame()
        top_cat = cat.sort_values("best_rank").head(1)
        row = {
            "dataset_key": dataset_key,
            "n_lr": int(len(sub)),
            "n_true_lr": int(sub["true_lr"].sum()),
            "n_high_expression_negative": int(sub["false_positive_high_expression"].sum()),
            "n_receptor_shared_risk": int(sub["receptor_shared_risk"].sum()),
            "n_ligand_shared_risk": int(sub["ligand_shared_risk"].sum()),
            "top10_true_lr": int(top10["n_true_lr"].iloc[0]) if len(top10) else 0,
            "top10_negative_or_risk_fraction": float(top10["negative_or_risk_fraction"].iloc[0]) if len(top10) else np.nan,
            "top10_control_panel_empirical_fdr": float(top10["control_panel_empirical_fdr"].iloc[0]) if len(top10) else np.nan,
            "top20_true_lr": int(top20["n_true_lr"].iloc[0]) if len(top20) else 0,
            "top20_negative_or_risk_fraction": float(top20["negative_or_risk_fraction"].iloc[0]) if len(top20) else np.nan,
            "top20_control_panel_empirical_fdr": float(top20["control_panel_empirical_fdr"].iloc[0]) if len(top20) else np.nan,
            "n_positive_top10": int(pos["top10"].sum()) if len(pos) else 0,
            "n_positive_rank_score_pass": int(pos["stress_pass_rank_score"].sum()) if len(pos) else 0,
            "n_positive_downstream_pass": int(pos["stress_pass_downstream"].sum()) if len(pos) else 0,
            "n_positive_empirical_fdr_0_1": int(pd.to_numeric(pos["empirical_bh_fdr_score"], errors="coerce").le(0.1).sum()) if len(pos) else 0,
            "median_positive_empirical_fdr": float(pd.to_numeric(pos["empirical_bh_fdr_score"], errors="coerce").median()) if len(pos) else np.nan,
            "dataset_negative_beat_fraction": float(stress["negative_comparison_beat_fraction"].iloc[0]) if len(stress) else np.nan,
            "dataset_receptor_shared_beats": int(stress["n_receptor_shared_beats"].iloc[0]) if len(stress) else 0,
            "n_modules": int(len(mod)),
            "n_negative_control_heavy_modules": int(mod["module_warning"].astype(str).str.contains("negative", case=False, na=False).sum()) if len(mod) else 0,
            "n_priority_anchor_enriched_modules": int(mod["module_warning"].astype(str).str.contains("priority", case=False, na=False).sum()) if len(mod) else 0,
            "top_biological_category_by_rank": str(top_cat["biological_category"].iloc[0]) if len(top_cat) else "",
            "top_category_first_lr": str(top_cat["top_lr_pair"].iloc[0]) if len(top_cat) else "",
        }
        row["atlas_audit_call"] = classify_dataset(row)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("dataset_key")


def classify_dataset(row: dict[str, object]) -> str:
    if (
        row["top10_true_lr"] >= 5
        and row["top10_negative_or_risk_fraction"] <= 0.35
        and row["dataset_negative_beat_fraction"] <= 0.2
    ):
        return "best_current_pilot_support_with_residual_risk"
    if row["top10_true_lr"] >= 3 and row["dataset_negative_beat_fraction"] <= 0.3:
        return "rank_support_with_negative_risk"
    if row["dataset_negative_beat_fraction"] > 0.35 or row["top10_negative_or_risk_fraction"] > 0.5:
        return "stress_failure_or_high_leakage"
    return "incomplete_or_mixed"


def write_status(dataset: pd.DataFrame, topk: pd.DataFrame, category: pd.DataFrame, positive: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Atlas FDR / Contamination Audit",
        "",
        "## Scope",
        "",
        "This summarizes the Phase 12 SpatialSelf atlas pilot. The empirical FDR values are based on hard-negative/control-panel ranks, not a final permutation-calibrated whole-atlas FDR.",
        "",
        "## Dataset Summary",
        "",
    ]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.atlas_audit_call}`: top10 true LR={int(row.top10_true_lr)}, "
            f"top10 risk fraction={fmt(row.top10_negative_or_risk_fraction)}, "
            f"top10 control-panel empirical FDR={fmt(row.top10_control_panel_empirical_fdr)}, "
            f"negative beat fraction={fmt(row.dataset_negative_beat_fraction)}, "
            f"receptor-shared beats={int(row.dataset_receptor_shared_beats)}, "
            f"modules={int(row.n_modules)}"
        )
    lines += ["", "## Top-K Contamination", ""]
    for row in topk[topk["top_k"].isin([10, 20])].itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` top{int(row.top_k)}: true={int(row.n_true_lr)}, "
            f"high-expression negatives={int(row.n_high_expression_negative)}, "
            f"receptor-risk={int(row.n_receptor_shared_risk)}, "
            f"risk fraction={fmt(row.negative_or_risk_fraction)}, "
            f"top category={row.top_biological_category}"
        )
    lines += ["", "## Positive-Control Stress", ""]
    for dataset_key, sub in positive.groupby("dataset_key"):
        top = sub.sort_values(["empirical_bh_fdr_score", "rank"], ascending=[True, True]).head(5)
        pieces = "; ".join(
            f"{r.lr_pair} rank={int(r.rank)} q={fmt(r.empirical_bh_fdr_score)} negBeat={fmt(r.negative_beat_fraction)}"
            for r in top.itertuples(index=False)
        )
        lines.append(f"- `{dataset_key}`: {pieces}")
    lines += [
        "",
        "## Claim Boundary",
        "",
        "- Use this as a pilot atlas audit, not final global FDR control.",
        "- A high top-k true-LR count is only useful when paired with low hard-negative and receptor-shared leakage.",
        "- Datasets marked stress/failure should remain limitations or calibration targets unless rerun with stronger null calibration.",
    ]
    (OUT_DIR / "phase12_atlas_fdr_audit_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest() -> None:
    manifest = {
        "task": "phase12_atlas_fdr_audit",
        "inputs": [
            "outputs/phase12/atlas_model_pilot/phase12_atlas_model_pilot_lr_ranking.csv",
            "outputs/phase12/hard_negative_stress/phase12_hard_negative_stress_positive_summary.csv",
            "outputs/phase12/hard_negative_stress/phase12_hard_negative_stress_dataset_summary.csv",
            "outputs/phase12/atlas_model_pilot/phase12_atlas_model_pilot_modules.csv",
        ],
        "outputs": [
            "phase12_atlas_fdr_dataset_summary.csv",
            "phase12_atlas_fdr_topk_summary.csv",
            "phase12_atlas_fdr_category_summary.csv",
            "phase12_atlas_fdr_positive_control_summary.csv",
            "phase12_atlas_fdr_audit_status.md",
        ],
        "claim_boundary": "Control-panel empirical FDR/contamination audit for pilot atlas runs; not final permutation-calibrated whole-atlas FDR.",
    }
    (OUT_DIR / "phase12_atlas_fdr_audit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def mode_or_empty(values: pd.Series) -> str:
    if values.empty:
        return ""
    counts = values.astype(str).value_counts()
    return str(counts.index[0]) if len(counts) else ""


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return np.nan
    return float(numerator) / float(denominator)


def fmt(value: object) -> str:
    try:
        if value is None or pd.isna(value):
            return "NA"
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(val) >= 100:
        return f"{val:.0f}"
    if abs(val) >= 10:
        return f"{val:.2f}"
    return f"{val:.3g}"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


if __name__ == "__main__":
    main()
