#!/usr/bin/env python
"""Same-universe baseline audit for Phase 12 expanded atlas runs.

This audit scores the same expanded LR universe and the same candidate edges as
the calibrated SpatialSelf atlas run using transparent heuristic baselines. The
goal is not to replace official COMMOT/CellChat/CellNEST/SpaTalk runs; it is to
answer a narrower fairness question:

Do simple expression, distance, expression-distance, or rule-prior scores rank
the same expanded LR universe as cleanly as calibrated SpatialSelf?

Official adapter-light multi-method baselines are reported as status rows when
available. Fully tuned end-to-end official runs remain a separate requirement.
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
OUT_DIR = EXPANDED_DIR / "same_universe_baselines"
TOP_K_VALUES = (10, 20, 50, 100)


METHOD_LABELS = {
    "spatialself_calibrated": "SpatialSelf calibrated",
    "spatialself_raw_mean": "SpatialSelf raw mean",
    "coexpression_distance_mean": "LR coexpression / distance",
    "coexpression_distance_q95": "LR coexpression / distance q95",
    "coexpression_only_mean": "LR coexpression only",
    "expression_support_fraction": "Nonzero LR edge fraction",
    "high_expression_naive": "High-expression naive",
    "distance_rule_prior": "Distance/type rule prior",
    "rule_expression_distance": "Rule-gated expression-distance",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    calibrated = pd.read_csv(EXPANDED_DIR / "phase12_atlas_model_expanded_calibrated_lr_ranking.csv")
    all_rankings = []
    all_topk = []
    all_hard = []
    official_status = official_baseline_status()
    for dataset_key in sorted(calibrated["dataset_key"].dropna().astype(str).unique()):
        replicate_dir = EXPANDED_DIR / "replicates" / dataset_key
        candidate = pd.read_csv(replicate_dir / "candidate_interactions.csv")
        lr_pairs = pd.read_csv(replicate_dir / "lr_pairs.csv")
        dataset_cal = calibrated[calibrated["dataset_key"].eq(dataset_key)].copy()
        ranking = build_dataset_rankings(dataset_key, candidate, lr_pairs, dataset_cal)
        topk = summarize_topk(ranking)
        hard = summarize_hard_negative(ranking)
        all_rankings.append(ranking)
        all_topk.append(topk)
        all_hard.append(hard)

    ranking_table = pd.concat(all_rankings, ignore_index=True) if all_rankings else pd.DataFrame()
    topk_table = pd.concat(all_topk, ignore_index=True) if all_topk else pd.DataFrame()
    hard_table = pd.concat(all_hard, ignore_index=True) if all_hard else pd.DataFrame()
    dataset_summary = summarize_dataset(topk_table, hard_table, ranking_table)
    method_summary = summarize_method(topk_table, hard_table, ranking_table)

    ranking_table.to_csv(OUT_DIR / "phase12_expanded_same_universe_baseline_lr_ranking.csv", index=False)
    topk_table.to_csv(OUT_DIR / "phase12_expanded_same_universe_baseline_topk_summary.csv", index=False)
    hard_table.to_csv(OUT_DIR / "phase12_expanded_same_universe_baseline_hard_negative.csv", index=False)
    dataset_summary.to_csv(OUT_DIR / "phase12_expanded_same_universe_baseline_dataset_summary.csv", index=False)
    method_summary.to_csv(OUT_DIR / "phase12_expanded_same_universe_baseline_method_summary.csv", index=False)
    official_status.to_csv(OUT_DIR / "phase12_expanded_official_baseline_status.csv", index=False)
    write_status(dataset_summary, method_summary, official_status)
    write_manifest()

    print(dataset_summary.to_string(index=False))
    print()
    print(method_summary.to_string(index=False))


def build_dataset_rankings(
    dataset_key: str,
    candidates: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    calibrated: pd.DataFrame,
) -> pd.DataFrame:
    table = candidates.copy()
    expr_product = np.log1p(pd.to_numeric(table["ligand_expr"], errors="coerce").fillna(0).to_numpy(dtype=float)) * np.log1p(
        pd.to_numeric(table["receptor_expr"], errors="coerce").fillna(0).to_numpy(dtype=float)
    )
    distance = pd.to_numeric(table["distance"], errors="coerce").fillna(0).to_numpy(dtype=float)
    type_rule = table["type_rule_satisfied"].astype(bool).to_numpy()
    range_rule = table["range_rule_satisfied"].astype(bool).to_numpy()
    table["_expr_product"] = expr_product
    table["_high_expression_sum"] = np.log1p(pd.to_numeric(table["ligand_expr"], errors="coerce").fillna(0).to_numpy(dtype=float)) + np.log1p(
        pd.to_numeric(table["receptor_expr"], errors="coerce").fillna(0).to_numpy(dtype=float)
    )
    table["_inverse_distance"] = 1.0 / (1.0 + distance)
    table["_rule_gate"] = (type_rule & range_rule).astype(float)
    table["_rule_expression_distance"] = table["_expr_product"] * table["_inverse_distance"] * table["_rule_gate"]
    table["_coexpression_distance"] = pd.to_numeric(table["coexpression_distance_score"], errors="coerce").fillna(0)

    grouped = (
        table.groupby("lr_id", as_index=False)
        .agg(
            coexpression_distance_mean=("_coexpression_distance", "mean"),
            coexpression_distance_q95=("_coexpression_distance", lambda x: float(np.quantile(x, 0.95))),
            coexpression_only_mean=("_expr_product", "mean"),
            expression_support_fraction=("_expr_product", lambda x: float((x > 0).mean())),
            high_expression_naive=("_high_expression_sum", "mean"),
            distance_rule_prior=("_rule_gate", "mean"),
            rule_expression_distance=("_rule_expression_distance", "mean"),
            n_candidates=("_expr_product", "size"),
            n_nonzero_expr=("_expr_product", lambda x: int((x > 0).sum())),
        )
        .merge(lr_pairs, on="lr_id", how="left")
    )
    grouped["lr_pair"] = grouped["ligand"] + "-" + grouped["receptor"]
    grouped["dataset_key"] = dataset_key

    spatialself = calibrated[
        [
            "dataset_key",
            "lr_id",
            "lr_pair",
            "calibrated_score",
            "raw_mean_score",
            "raw_q95_score",
            "rank",
            "true_lr",
            "false_positive_high_expression",
            "biological_category",
            "mechanism",
            "ligand",
            "receptor",
            "n_nonzero_expr",
        ]
    ].copy()
    spatialself["method"] = "spatialself_calibrated"
    spatialself["method_label"] = METHOD_LABELS["spatialself_calibrated"]
    spatialself["score"] = pd.to_numeric(spatialself["calibrated_score"], errors="coerce").fillna(0)
    spatialself["rank"] = spatialself["score"].rank(ascending=False, method="min").astype(int)
    spatialself["score_source"] = "calibrated SpatialSelf score"

    raw = spatialself.copy()
    raw["method"] = "spatialself_raw_mean"
    raw["method_label"] = METHOD_LABELS["spatialself_raw_mean"]
    raw["score"] = pd.to_numeric(raw["raw_mean_score"], errors="coerce").fillna(0)
    raw["rank"] = raw["score"].rank(ascending=False, method="min").astype(int)
    raw["score_source"] = "raw SpatialSelf mean score"

    rows = [spatialself, raw]
    baseline_score_cols = [
        "coexpression_distance_mean",
        "coexpression_distance_q95",
        "coexpression_only_mean",
        "expression_support_fraction",
        "high_expression_naive",
        "distance_rule_prior",
        "rule_expression_distance",
    ]
    for method in baseline_score_cols:
        part = grouped.copy()
        part["method"] = method
        part["method_label"] = METHOD_LABELS[method]
        part["score"] = pd.to_numeric(part[method], errors="coerce").fillna(0)
        part["rank"] = part["score"].rank(ascending=False, method="min").astype(int)
        part["score_source"] = method
        rows.append(part)

    out = pd.concat(rows, ignore_index=True, sort=False)
    out = annotate_risk(out)
    keep = [
        "dataset_key",
        "method",
        "method_label",
        "score_source",
        "rank",
        "score",
        "lr_pair",
        "lr_id",
        "ligand",
        "receptor",
        "biological_category",
        "mechanism",
        "true_lr",
        "false_positive_high_expression",
        "receptor_shared_risk",
        "ligand_shared_risk",
        "negative_or_risk",
        "n_candidates",
        "n_nonzero_expr",
    ]
    return out[keep].sort_values(["dataset_key", "method", "rank", "lr_pair"]).reset_index(drop=True)


def annotate_risk(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["true_lr"] = out["true_lr"].astype(bool)
    out["false_positive_high_expression"] = out["false_positive_high_expression"].astype(bool)
    out["receptor_shared_risk"] = False
    out["ligand_shared_risk"] = False
    for (dataset_key, method), sub in out.groupby(["dataset_key", "method"], sort=False):
        true_receptors = set(sub[sub["true_lr"]]["receptor"].astype(str))
        true_ligands = set(sub[sub["true_lr"]]["ligand"].astype(str))
        mask = out["dataset_key"].eq(dataset_key) & out["method"].eq(method)
        out.loc[mask, "receptor_shared_risk"] = (
            out.loc[mask, "receptor"].astype(str).isin(true_receptors) & ~out.loc[mask, "true_lr"].astype(bool)
        )
        out.loc[mask, "ligand_shared_risk"] = (
            out.loc[mask, "ligand"].astype(str).isin(true_ligands) & ~out.loc[mask, "true_lr"].astype(bool)
        )
    out["negative_or_risk"] = out["false_positive_high_expression"] | out["receptor_shared_risk"] | out["ligand_shared_risk"]
    return out


def summarize_topk(ranking: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset_key, method), sub in ranking.groupby(["dataset_key", "method"], sort=False):
        sub = sub.sort_values("rank")
        for top_k in TOP_K_VALUES:
            top = sub.head(top_k)
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "method": method,
                    "method_label": top["method_label"].iloc[0],
                    "top_k": top_k,
                    "n_lr": int(len(top)),
                    "n_true_lr": int(top["true_lr"].astype(bool).sum()),
                    "true_lr_fraction": safe_div(top["true_lr"].astype(bool).sum(), len(top)),
                    "n_high_expression_negative": int(top["false_positive_high_expression"].astype(bool).sum()),
                    "n_receptor_shared_risk": int(top["receptor_shared_risk"].sum()),
                    "n_ligand_shared_risk": int(top["ligand_shared_risk"].sum()),
                    "n_any_negative_or_risk": int(top["negative_or_risk"].sum()),
                    "negative_or_risk_fraction": safe_div(top["negative_or_risk"].sum(), len(top)),
                    "control_panel_empirical_fdr": safe_div(
                        top["negative_or_risk"].sum(),
                        max(int(top["negative_or_risk"].sum()) + int(top["true_lr"].astype(bool).sum()), 1),
                    ),
                    "top_biological_category": mode_or_empty(top["biological_category"]),
                    "top_lr_pairs": ";".join(top["lr_pair"].astype(str).head(12)),
                }
            )
    return pd.DataFrame(rows)


def summarize_hard_negative(ranking: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset_key, method), sub in ranking.groupby(["dataset_key", "method"], sort=False):
        positives = sub[sub["true_lr"].astype(bool)].copy()
        negatives = sub[sub["false_positive_high_expression"].astype(bool)].copy()
        for pos in positives.itertuples(index=False):
            pools = [
                ("same_receptor", negatives[negatives["receptor"].astype(str).eq(str(pos.receptor))]),
                ("same_ligand", negatives[negatives["ligand"].astype(str).eq(str(pos.ligand))]),
                ("any_high_expression_negative", negatives),
            ]
            for negative_type, pool in pools:
                if pool.empty:
                    continue
                neg = pool.sort_values(["rank", "score"], ascending=[True, False]).iloc[0]
                rows.append(
                    {
                        "dataset_key": dataset_key,
                        "method": method,
                        "method_label": pos.method_label,
                        "positive_lr_pair": pos.lr_pair,
                        "positive_rank": int(pos.rank),
                        "positive_score": float(pos.score),
                        "negative_pair": neg["lr_pair"],
                        "negative_type": negative_type,
                        "negative_rank": int(neg["rank"]),
                        "negative_score": float(neg["score"]),
                        "negative_beats_positive": bool(int(neg["rank"]) < int(pos.rank)),
                    }
                )
    return pd.DataFrame(rows)


def summarize_dataset(topk: pd.DataFrame, hard: pd.DataFrame, ranking: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset_key, method), sub in ranking.groupby(["dataset_key", "method"], sort=False):
        top10 = one_row(topk, dataset_key, method, 10)
        top50 = one_row(topk, dataset_key, method, 50)
        hard_sub = hard[hard["dataset_key"].eq(dataset_key) & hard["method"].eq(method)] if not hard.empty else pd.DataFrame()
        rows.append(
            {
                "dataset_key": dataset_key,
                "method": method,
                "method_label": sub["method_label"].iloc[0],
                "n_lr": int(len(sub)),
                "top10_true_lr": int(top10.get("n_true_lr", 0)),
                "top10_high_expression_negative": int(top10.get("n_high_expression_negative", 0)),
                "top10_negative_or_risk_fraction": float(top10.get("negative_or_risk_fraction", np.nan)),
                "top10_control_panel_empirical_fdr": float(top10.get("control_panel_empirical_fdr", np.nan)),
                "top50_true_lr": int(top50.get("n_true_lr", 0)),
                "top50_high_expression_negative": int(top50.get("n_high_expression_negative", 0)),
                "top50_negative_or_risk_fraction": float(top50.get("negative_or_risk_fraction", np.nan)),
                "hard_negative_best_comparison_beats": int(hard_sub["negative_beats_positive"].sum()) if len(hard_sub) else 0,
                "hard_negative_best_comparisons": int(len(hard_sub)),
                "hard_negative_best_comparison_beat_fraction": safe_div(
                    int(hard_sub["negative_beats_positive"].sum()) if len(hard_sub) else 0,
                    len(hard_sub),
                ),
                "best_true_lr": best_true_lr(sub),
                "best_true_lr_rank": best_true_lr_rank(sub),
                "same_universe_call": classify_method(top10, hard_sub),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset_key", "top10_true_lr", "top10_negative_or_risk_fraction"], ascending=[True, False, True])


def summarize_method(topk: pd.DataFrame, hard: pd.DataFrame, ranking: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dataset = summarize_dataset(topk, hard, ranking)
    for method, sub in dataset.groupby("method", sort=False):
        rows.append(
            {
                "method": method,
                "method_label": sub["method_label"].iloc[0],
                "n_datasets": int(len(sub)),
                "mean_top10_true_lr": float(sub["top10_true_lr"].mean()),
                "mean_top10_high_expression_negative": float(sub["top10_high_expression_negative"].mean()),
                "mean_top10_negative_or_risk_fraction": float(sub["top10_negative_or_risk_fraction"].mean()),
                "mean_hard_negative_beat_fraction": float(sub["hard_negative_best_comparison_beat_fraction"].mean()),
                "n_clean_top10_datasets": int(
                    (
                        sub["top10_high_expression_negative"].eq(0)
                        & sub["top10_negative_or_risk_fraction"].le(0.25)
                        & sub["top10_true_lr"].ge(3)
                    ).sum()
                ),
                "summary_call": summarize_method_call(sub),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["n_clean_top10_datasets", "mean_top10_true_lr", "mean_top10_negative_or_risk_fraction"],
        ascending=[False, False, True],
    )


def official_baseline_status() -> pd.DataFrame:
    phase12_adapter = PROJECT_ROOT / "outputs/phase12/atlas_model_expanded/official_adapter_baselines/phase12_expanded_official_adapter_status.csv"
    if phase12_adapter.exists():
        current = pd.read_csv(phase12_adapter).copy()
        current["dataset_scope"] = "phase12_expanded_886_957_lr_adapter_light"
        current["evidence_file"] = rel(phase12_adapter)
        return current[
            [
                "method",
                "dataset_key",
                "dataset_scope",
                "status",
                "detail",
                "same_universe_phase12_available",
                "usable_in_phase12_expanded_claim",
                "evidence_file",
            ]
        ]
    rows = []
    phase6 = PROJECT_ROOT / "outputs/phase6/expanded_lr_universe/baselines/lymph_node_phase6_expanded/baseline_status.csv"
    if phase6.exists():
        old = pd.read_csv(phase6)
        for row in old.itertuples(index=False):
            rows.append(
                {
                    "method": row.method,
                    "dataset_scope": "phase6_lymph_node_expanded_not_phase12_957lr",
                    "status": row.status,
                    "detail": row.detail,
                    "same_universe_phase12_available": False,
                    "usable_in_phase12_expanded_claim": "no_direct",
                    "evidence_file": rel(phase6),
                }
            )
    for method in ["COMMOT", "CellChat", "SpaTalk", "CellNEST"]:
        if not any(str(r.get("method", "")).lower() == method.lower() for r in rows):
            rows.append(
                {
                    "method": method,
                    "dataset_scope": "phase12_expanded_886_957_lr",
                    "status": "not_run_same_universe",
                    "detail": "No official same-universe expanded Phase 12 run found for post-Xenium/human lymph node calibrated atlas.",
                    "same_universe_phase12_available": False,
                    "usable_in_phase12_expanded_claim": "no",
                    "evidence_file": "",
                }
            )
    return pd.DataFrame(rows)


def classify_method(top10: dict[str, object], hard_sub: pd.DataFrame) -> str:
    top10_true = int(top10.get("n_true_lr", 0))
    top10_high = int(top10.get("n_high_expression_negative", 0))
    risk = float(top10.get("negative_or_risk_fraction", np.nan))
    beat_fraction = safe_div(int(hard_sub["negative_beats_positive"].sum()) if len(hard_sub) else 0, len(hard_sub))
    if top10_true >= 3 and top10_high == 0 and risk <= 0.25 and beat_fraction <= 0.35:
        return "same_universe_topk_support"
    if top10_true >= 3 and top10_high == 0:
        return "same_universe_rank_support_with_hard_negative_risk"
    if top10_high > 0 or risk >= 0.5:
        return "same_universe_contaminated"
    return "same_universe_mixed"


def summarize_method_call(sub: pd.DataFrame) -> str:
    clean = int(
        (
            sub["top10_high_expression_negative"].eq(0)
            & sub["top10_negative_or_risk_fraction"].le(0.25)
            & sub["top10_true_lr"].ge(3)
        ).sum()
    )
    if clean == len(sub):
        return "clean_top10_across_expanded_datasets"
    if clean:
        return "partial_clean_top10"
    if sub["top10_high_expression_negative"].mean() > 0 or sub["top10_negative_or_risk_fraction"].mean() >= 0.5:
        return "contaminated_or_risk_prone"
    return "mixed_or_low_true_lr"


def write_status(dataset: pd.DataFrame, method: pd.DataFrame, official: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Expanded Same-Universe Baseline Audit",
        "",
        "## Scope",
        "",
        "This audit compares calibrated SpatialSelf with transparent heuristic baselines on the exact same expanded LR universe and candidate edges.",
        "It is a heuristic same-universe audit; adapter-light COMMOT/CellChat/SpaTalk/CellNEST status is reported separately when available.",
        "",
        "## Method Summary",
        "",
    ]
    for row in method.itertuples(index=False):
        lines.append(
            f"- `{row.method}` `{row.summary_call}`: mean top10 true={fmt(row.mean_top10_true_lr)}, "
            f"mean high-neg={fmt(row.mean_top10_high_expression_negative)}, "
            f"mean risk={fmt(row.mean_top10_negative_or_risk_fraction)}, "
            f"mean hard-neg beat={fmt(row.mean_hard_negative_beat_fraction)}, "
            f"clean datasets={int(row.n_clean_top10_datasets)}/{int(row.n_datasets)}"
        )
    lines += ["", "## Dataset Detail", ""]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.method}` `{row.same_universe_call}`: "
            f"top10 true={int(row.top10_true_lr)}, high-neg={int(row.top10_high_expression_negative)}, "
            f"risk={fmt(row.top10_negative_or_risk_fraction)}, hard-neg beat={int(row.hard_negative_best_comparison_beats)}/{int(row.hard_negative_best_comparisons)}, "
            f"best true={row.best_true_lr}#{int(row.best_true_lr_rank)}"
        )
    lines += ["", "## Official Baseline Status", ""]
    for row in official.itertuples(index=False):
        lines.append(
            f"- `{row.method}`: {row.status}; scope={row.dataset_scope}; same-universe Phase12={bool(row.same_universe_phase12_available)}; {row.detail}"
        )
    lines += [
        "",
        "## Claim Boundary",
        "",
        "- Use this to compare calibrated SpatialSelf against simple same-universe scoring baselines.",
        "- Adapter-light COMMOT/CellChat/SpaTalk/CellNEST expanded runs are reported in the status table when available.",
        "- Do not present adapter-light projected outputs as fully tuned official end-to-end baselines.",
    ]
    (OUT_DIR / "phase12_expanded_same_universe_baseline_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest() -> None:
    manifest = {
        "task": "phase12_expanded_same_universe_baselines",
        "inputs": [
            "outputs/phase12/atlas_model_expanded/phase12_atlas_model_expanded_calibrated_lr_ranking.csv",
            "outputs/phase12/atlas_model_expanded/replicates/*/candidate_interactions.csv",
            "outputs/phase12/atlas_model_expanded/replicates/*/lr_pairs.csv",
        ],
        "outputs": [
            "phase12_expanded_same_universe_baseline_lr_ranking.csv",
            "phase12_expanded_same_universe_baseline_topk_summary.csv",
            "phase12_expanded_same_universe_baseline_hard_negative.csv",
            "phase12_expanded_same_universe_baseline_dataset_summary.csv",
            "phase12_expanded_same_universe_baseline_method_summary.csv",
            "phase12_expanded_official_baseline_status.csv",
            "phase12_expanded_same_universe_baseline_status.md",
        ],
        "claim_boundary": "Same-universe heuristic baseline audit; adapter-light official method projections are separate from fully tuned end-to-end baselines.",
    }
    (OUT_DIR / "phase12_expanded_same_universe_baseline_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def best_true_lr(sub: pd.DataFrame) -> str:
    true = sub[sub["true_lr"].astype(bool)].sort_values("rank")
    return str(true.iloc[0]["lr_pair"]) if len(true) else ""


def best_true_lr_rank(sub: pd.DataFrame) -> int:
    true = sub[sub["true_lr"].astype(bool)].sort_values("rank")
    return int(true.iloc[0]["rank"]) if len(true) else -1


def one_row(table: pd.DataFrame, dataset_key: str, method: str, top_k: int) -> dict[str, object]:
    sub = table[table["dataset_key"].eq(dataset_key) & table["method"].eq(method) & table["top_k"].eq(top_k)]
    if sub.empty:
        return {}
    return sub.iloc[0].to_dict()


def mode_or_empty(values: pd.Series) -> str:
    if values.empty:
        return ""
    counts = values.astype(str).value_counts()
    return str(counts.index[0]) if len(counts) else ""


def safe_div(num: float, denom: float) -> float:
    return float(num) / float(denom) if float(denom) else float("nan")


def fmt(value: object) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(val):
        return "NA"
    return f"{val:.3g}"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
