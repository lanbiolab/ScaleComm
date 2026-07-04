#!/usr/bin/env python
"""Edge-level hard-negative audit for Phase 12 expanded atlas predictions.

The LR-pair hard-negative stress test checks same-receptor and high-expression
false LR pairs. This audit adds fixed-score edge-level controls: wrong spatial
range, wrong niche, wrong cell-type/rule, zero-expression edges, and same
sender-receiver edges carrying non-anchor LR identities.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


ATLAS_DIR = PROJECT_ROOT / "outputs/phase12/atlas_model_expanded"
OUT_DIR = PROJECT_ROOT / "outputs/phase12/edge_hard_negative_audit"
TOP_FRACTION = 0.10

DATASETS = {
    "post_xenium_colon_full": {
        "replicate_dir": ATLAS_DIR / "replicates/post_xenium_colon_full",
        "prediction": ATLAS_DIR
        / "predictions/post_xenium_colon_full/spatialself_wavelet_edge_prior/"
        "post_xenium_colon_full__spatialself_wavelet_predictions.csv",
    },
    "human_lymph_node_visium_raw_full": {
        "replicate_dir": ATLAS_DIR / "replicates/human_lymph_node_visium_raw_full",
        "prediction": ATLAS_DIR
        / "predictions/human_lymph_node_visium_raw_full/spatialself_wavelet_edge_prior/"
        "human_lymph_node_visium_raw_full__spatialself_wavelet_predictions.csv",
    },
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_lr_rows = []
    all_class_rows = []
    for dataset_key, spec in DATASETS.items():
        lr_rows, class_rows = audit_dataset(dataset_key, Path(spec["replicate_dir"]), Path(spec["prediction"]))
        all_lr_rows.append(lr_rows)
        all_class_rows.append(class_rows)
    lr_table = pd.concat(all_lr_rows, ignore_index=True) if all_lr_rows else pd.DataFrame()
    class_table = pd.concat(all_class_rows, ignore_index=True) if all_class_rows else pd.DataFrame()
    dataset_summary = summarize_datasets(class_table)

    lr_table.to_csv(OUT_DIR / "phase12_edge_hard_negative_lr_summary.csv", index=False)
    class_table.to_csv(OUT_DIR / "phase12_edge_hard_negative_class_summary.csv", index=False)
    dataset_summary.to_csv(OUT_DIR / "phase12_edge_hard_negative_dataset_summary.csv", index=False)
    write_status(dataset_summary, class_table)
    write_manifest(lr_table, class_table, dataset_summary)
    print(dataset_summary.to_string(index=False))
    print()
    print(class_table.to_string(index=False))


def audit_dataset(dataset_key: str, replicate_dir: Path, prediction_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
    lr_pairs = pd.read_csv(replicate_dir / "lr_pairs.csv")
    prediction = pd.read_csv(prediction_path)
    table = candidates.merge(prediction[["candidate_id", "score"]], on="candidate_id", how="inner", validate="one_to_one")
    table["score"] = pd.to_numeric(table["score"], errors="coerce").fillna(0.0)
    table["raw_score"] = table["score"]
    table["edge_guardrail_score"] = table["score"] * edge_guardrail_weight(table)
    table["edge_pair"] = table["sender_id"].astype(str) + "->" + table["receiver_id"].astype(str)
    true_lr = lr_pairs[lr_pairs.get("true_lr", pd.Series(False, index=lr_pairs.index)).fillna(False).astype(bool)]

    lr_rows = []
    for score_mode, score_column in [("raw", "raw_score"), ("edge_guardrail", "edge_guardrail_score")]:
        mode_table = table.copy()
        mode_table["score"] = mode_table[score_column]
        for lr in true_lr.itertuples(index=False):
            lr_sub = mode_table[mode_table["lr_id"].astype(str).eq(str(lr.lr_id))].copy()
            if lr_sub.empty:
                continue
            positive = lr_sub[
                lr_sub["true_lr"].fillna(False).astype(bool)
                & lr_sub["type_rule_satisfied"].fillna(False).astype(bool)
                & lr_sub["range_rule_satisfied"].fillna(False).astype(bool)
                & lr_sub["niche_rule_satisfied"].fillna(False).astype(bool)
                & (pd.to_numeric(lr_sub["ligand_expr"], errors="coerce").fillna(0) > 0)
                & (pd.to_numeric(lr_sub["receptor_expr"], errors="coerce").fillna(0) > 0)
            ]
            if positive.empty:
                positive = lr_sub[lr_sub["true_lr"].fillna(False).astype(bool)]
            pos_stat = score_summary(positive)
            pos_edge_pairs = set(
                positive.sort_values("score", ascending=False).head(max(10, len(positive) // 10))["edge_pair"].astype(str)
            )
            class_masks = negative_class_masks(lr_sub, mode_table, pos_edge_pairs, str(lr.lr_id))
            for negative_class, negative in class_masks.items():
                neg_stat = score_summary(negative)
                lr_rows.append(
                    {
                        "dataset_key": dataset_key,
                        "score_mode": score_mode,
                        "lr_id": lr.lr_id,
                        "lr_pair": f"{lr.ligand}-{lr.receptor}",
                        "biological_category": getattr(lr, "biological_category", ""),
                        "negative_class": negative_class,
                        **prefix_dict(pos_stat, "positive"),
                        **prefix_dict(neg_stat, "negative"),
                        "negative_minus_positive_top_tail": neg_stat["top_tail_mean"] - pos_stat["top_tail_mean"],
                        "negative_leakage": bool(neg_stat["top_tail_mean"] >= pos_stat["top_tail_mean"]) if neg_stat["n_edges"] else False,
                        "empirical_p_negative_ge_positive": empirical_p(negative["score"], pos_stat["top_tail_mean"]),
                    }
                )
    lr_table = pd.DataFrame(lr_rows)
    class_summary = summarize_classes(dataset_key, lr_table)
    return lr_table, class_summary


def edge_guardrail_weight(table: pd.DataFrame) -> np.ndarray:
    ligand = pd.to_numeric(table.get("ligand_expr", pd.Series(0, index=table.index)), errors="coerce").fillna(0).to_numpy()
    receptor = pd.to_numeric(table.get("receptor_expr", pd.Series(0, index=table.index)), errors="coerce").fillna(0).to_numpy()
    both_present = (ligand > 0) & (receptor > 0)
    one_missing = (ligand > 0) ^ (receptor > 0)
    weight = np.where(both_present, 1.0, np.where(one_missing, 0.12, 0.03))
    for column, penalty in [
        ("type_rule_satisfied", 0.35),
        ("range_rule_satisfied", 0.20),
        ("niche_rule_satisfied", 0.30),
    ]:
        if column in table:
            weight *= np.where(table[column].fillna(False).astype(bool).to_numpy(), 1.0, penalty)
    if "false_positive_high_expression" in table:
        weight *= np.where(table["false_positive_high_expression"].fillna(False).astype(bool).to_numpy(), 0.10, 1.0)
    return np.clip(weight, 0.0, 1.0)


def negative_class_masks(
    lr_sub: pd.DataFrame,
    all_edges: pd.DataFrame,
    positive_edge_pairs: set[str],
    lr_id: str,
) -> dict[str, pd.DataFrame]:
    ligand = pd.to_numeric(lr_sub["ligand_expr"], errors="coerce").fillna(0)
    receptor = pd.to_numeric(lr_sub["receptor_expr"], errors="coerce").fillna(0)
    far_threshold = float(pd.to_numeric(lr_sub["distance"], errors="coerce").fillna(0).quantile(0.90))
    same_spatial_pairs = all_edges[
        all_edges["edge_pair"].astype(str).isin(positive_edge_pairs)
        & ~all_edges["lr_id"].astype(str).eq(str(lr_id))
        & ~all_edges["true_lr"].fillna(False).astype(bool)
    ]
    return {
        "same_lr_wrong_range": lr_sub[~lr_sub["range_rule_satisfied"].fillna(False).astype(bool)],
        "same_lr_wrong_niche": lr_sub[~lr_sub["niche_rule_satisfied"].fillna(False).astype(bool)],
        "same_lr_wrong_type": lr_sub[~lr_sub["type_rule_satisfied"].fillna(False).astype(bool)],
        "same_lr_zero_expression": lr_sub[(ligand <= 0) | (receptor <= 0)],
        "same_lr_spatially_shifted_sender_proxy": lr_sub[
            pd.to_numeric(lr_sub["distance"], errors="coerce").fillna(0).ge(far_threshold)
        ],
        "same_spatial_edge_wrong_lr_identity": same_spatial_pairs,
        "high_expression_false_lr_edges": all_edges[all_edges["false_positive_high_expression"].fillna(False).astype(bool)],
    }


def score_summary(table: pd.DataFrame) -> dict[str, float]:
    if table.empty:
        return {
            "n_edges": 0,
            "mean_score": np.nan,
            "max_score": np.nan,
            "top_tail_mean": np.nan,
            "nonzero_expr_fraction": np.nan,
            "rule_satisfied_fraction": np.nan,
        }
    score = pd.to_numeric(table["score"], errors="coerce").fillna(0).to_numpy(dtype=float)
    n_top = max(int(np.ceil(len(score) * TOP_FRACTION)), 1)
    ligand = pd.to_numeric(table.get("ligand_expr", pd.Series(np.nan, index=table.index)), errors="coerce").fillna(0)
    receptor = pd.to_numeric(table.get("receptor_expr", pd.Series(np.nan, index=table.index)), errors="coerce").fillna(0)
    rule = (
        table.get("type_rule_satisfied", pd.Series(False, index=table.index)).fillna(False).astype(bool)
        & table.get("range_rule_satisfied", pd.Series(False, index=table.index)).fillna(False).astype(bool)
        & table.get("niche_rule_satisfied", pd.Series(False, index=table.index)).fillna(False).astype(bool)
    )
    return {
        "n_edges": int(len(table)),
        "mean_score": float(score.mean()),
        "max_score": float(score.max()),
        "top_tail_mean": float(np.sort(score)[-n_top:].mean()),
        "nonzero_expr_fraction": float(((ligand > 0) & (receptor > 0)).mean()),
        "rule_satisfied_fraction": float(rule.mean()),
    }


def summarize_classes(dataset_key: str, lr_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if lr_table.empty:
        return pd.DataFrame()
    for (score_mode, negative_class), sub in lr_table.groupby(["score_mode", "negative_class"], sort=False):
        valid = sub[pd.to_numeric(sub["negative_n_edges"], errors="coerce").fillna(0).gt(0)]
        rows.append(
            {
                "dataset_key": dataset_key,
                "score_mode": score_mode,
                "negative_class": negative_class,
                "n_lr_tested": int(len(sub)),
                "n_lr_with_negative_edges": int(len(valid)),
                "n_negative_leakage": int(valid["negative_leakage"].fillna(False).sum()),
                "negative_leakage_fraction": safe_divide(int(valid["negative_leakage"].fillna(False).sum()), len(valid)),
                "median_negative_minus_positive_top_tail": float(valid["negative_minus_positive_top_tail"].median()) if len(valid) else np.nan,
                "median_empirical_p": float(valid["empirical_p_negative_ge_positive"].median()) if len(valid) else np.nan,
                "hardest_lr_pair": hardest_lr(valid),
            }
        )
    return pd.DataFrame(rows)


def summarize_datasets(class_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if class_table.empty:
        return pd.DataFrame()
    for (dataset_key, score_mode), sub in class_table.groupby(["dataset_key", "score_mode"], sort=False):
        rows.append(
            {
                "dataset_key": dataset_key,
                "score_mode": score_mode,
                "n_negative_classes": int(sub["negative_class"].nunique()),
                "n_lr_class_tests": int(sub["n_lr_with_negative_edges"].sum()),
                "n_leakage_lr_class_tests": int(sub["n_negative_leakage"].sum()),
                "mean_leakage_fraction": float(sub["negative_leakage_fraction"].mean()),
                "hardest_negative_class": str(
                    sub.sort_values(["negative_leakage_fraction", "median_negative_minus_positive_top_tail"], ascending=False)
                    .iloc[0]["negative_class"]
                ),
                "interpretation": interpret_dataset(sub),
            }
        )
    return pd.DataFrame(rows)


def interpret_dataset(sub: pd.DataFrame) -> str:
    max_leak = float(sub["negative_leakage_fraction"].max()) if len(sub) else 0.0
    if max_leak >= 0.5:
        return "edge-level hard negatives expose substantial leakage"
    if max_leak > 0:
        return "some edge-level hard-negative leakage remains"
    return "no edge-level hard-negative class beats positive top-tail scores"


def hardest_lr(table: pd.DataFrame) -> str:
    if table.empty:
        return ""
    row = table.sort_values("negative_minus_positive_top_tail", ascending=False).iloc[0]
    return str(row["lr_pair"])


def empirical_p(negative_scores: pd.Series, positive_stat: float) -> float:
    scores = pd.to_numeric(negative_scores, errors="coerce").dropna().to_numpy(dtype=float)
    if len(scores) == 0 or not np.isfinite(positive_stat):
        return np.nan
    return float((1 + np.sum(scores >= positive_stat)) / (1 + len(scores)))


def prefix_dict(values: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def safe_divide(num: int, den: int) -> float:
    return float(num / den) if den else np.nan


def write_status(dataset_summary: pd.DataFrame, class_table: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Edge-Level Hard-Negative Audit",
        "",
        "## Scope",
        "",
        "This fixed-score audit uses expanded-atlas candidate-edge predictions to compare rule-satisfied positive edges against edge-level hard negatives: wrong range, wrong niche, wrong type, zero expression, spatially shifted sender proxy, same spatial edge with wrong LR identity, and high-expression false LR edges. It reports both raw model scores and a diagnostic edge_guardrail score that penalizes expression/rule violations.",
        "",
        "## Dataset Summary",
        "",
    ]
    for row in dataset_summary.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.score_mode}`: classes={int(row.n_negative_classes)}, "
            f"leakage tests={int(row.n_leakage_lr_class_tests)}/{int(row.n_lr_class_tests)}, "
            f"mean leakage={row.mean_leakage_fraction:.3g}, hardest={row.hardest_negative_class}; {row.interpretation}"
        )
    lines.extend(["", "## Hardest Classes", ""])
    for row in class_table.sort_values(["dataset_key", "score_mode", "negative_leakage_fraction"], ascending=[True, True, False]).itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.score_mode}` `{row.negative_class}`: leakage={int(row.n_negative_leakage)}/{int(row.n_lr_with_negative_edges)}, "
            f"median delta={row.median_negative_minus_positive_top_tail:.3g}, hardest LR={row.hardest_lr_pair}"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "This is a fixed-score edge-level stress audit, not a model-rerun null. The edge_guardrail score is a diagnostic post-hoc calibration, not a retrained final model. Use these results to identify leakage modes and calibration targets before manuscript-level false-positive control claims.",
        ]
    )
    (OUT_DIR / "phase12_edge_hard_negative_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(lr_table: pd.DataFrame, class_table: pd.DataFrame, dataset_summary: pd.DataFrame) -> None:
    manifest = {
        "task": "phase12_edge_hard_negative_audit",
        "datasets": sorted(dataset_summary["dataset_key"].dropna().astype(str).unique().tolist()) if not dataset_summary.empty else [],
        "top_fraction": TOP_FRACTION,
        "score_modes": sorted(dataset_summary["score_mode"].dropna().astype(str).unique().tolist()) if "score_mode" in dataset_summary else [],
        "n_lr_class_rows": int(len(lr_table)),
        "n_class_summary_rows": int(len(class_table)),
        "outputs": {
            "lr_summary": rel(OUT_DIR / "phase12_edge_hard_negative_lr_summary.csv"),
            "class_summary": rel(OUT_DIR / "phase12_edge_hard_negative_class_summary.csv"),
            "dataset_summary": rel(OUT_DIR / "phase12_edge_hard_negative_dataset_summary.csv"),
            "status": rel(OUT_DIR / "phase12_edge_hard_negative_status.md"),
        },
        "claim_boundary": "Fixed-score edge-level hard-negative stress audit; edge_guardrail is diagnostic post-hoc calibration, not retrained null FDR.",
    }
    (OUT_DIR / "phase12_edge_hard_negative_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
