#!/usr/bin/env python
"""Summarize the five-method native lymph node full-transcriptome panel."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "outputs/phase11/commot_style_spatial_story/human_lymph_node_visium_full_subset"
RESULT_DIR = OUT_DIR / "results"
BASELINE_DIR = OUT_DIR / "baselines"


FOCUS_LR = ["CCL19-CCR7", "CCL21-CCR7", "CXCL13-CXCR5", "CD48-CD244"]


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    impact = pd.read_csv(RESULT_DIR / "lymph_node_full_method_impact_summary.csv")
    direction = pd.read_csv(RESULT_DIR / "lymph_node_full_directionality_lr_summary.csv")
    modules = pd.read_csv(RESULT_DIR / "lymph_node_full_modules_summary.csv")
    metrics = pd.read_csv(BASELINE_DIR / "baseline_metrics_extended.csv")
    status = pd.read_csv(BASELINE_DIR / "baseline_status_extended.csv")
    metrics["method"] = metrics["method"].map(normalize_method).fillna(metrics["method"])
    status["method"] = status["method"].map(normalize_method).fillna(status["method"])

    rows = []
    for method in sorted(set(impact["method"]) | set(direction["method"]) | set(metrics["method"])):
        imp = first(impact[impact["method"].eq(method)])
        met = first(metrics[metrics["method"].eq(method)])
        stat = first(status[status["method"].eq(method)])
        mod = modules[modules["method"].eq(method)]
        true_clean_modules = mod[(mod["n_true_lr"].gt(0)) & (mod["n_high_expression_negative"].eq(0))]
        neg_modules = mod[mod["n_high_expression_negative"].gt(0)]
        rows.append(
            {
                "method": method,
                "status": value(stat, "status"),
                "baseline_top_lr_by_mean_score": value(met, "top_lr_by_mean_score"),
                "baseline_top_lr_is_high_expression_negative": is_negative_lr(value(met, "top_lr_by_mean_score")),
                "mean_priority_delta_r2": num(imp, "mean_priority_delta_r2"),
                "mean_weak_priority_delta_r2": num(imp, "mean_weak_priority_delta_r2"),
                "max_negative_delta_r2": num(imp, "max_negative_delta_r2"),
                "priority_minus_max_negative": num(imp, "priority_minus_max_negative"),
                "best_priority_lr": value(imp, "best_priority_lr"),
                "best_priority_delta_r2": num(imp, "best_priority_delta_r2"),
                "ccr7_flow_supported_methods_note": ccr7_note(direction, method),
                "ccl19_ccr7_expected_flow_fraction": flow(direction, method, "CCL19-CCR7"),
                "ccl21_ccr7_expected_flow_fraction": flow(direction, method, "CCL21-CCR7"),
                "cxcl13_cxcr5_expected_flow_fraction": flow(direction, method, "CXCL13-CXCR5"),
                "cd48_cd244_expected_flow_fraction": flow(direction, method, "CD48-CD244"),
                "n_modules": int(len(mod)),
                "n_clean_true_modules": int(len(true_clean_modules)),
                "n_negative_containing_modules": int(len(neg_modules)),
                "max_module_downstream_delta_r2": float(mod["mean_downstream_delta_r2"].max()) if len(mod) else np.nan,
                "interpretation": interpretation(method, imp, met, direction, mod),
            }
        )

    summary = pd.DataFrame(rows).sort_values(["status", "method"], ascending=[True, True])
    summary.to_csv(RESULT_DIR / "lymph_node_full_five_method_panel_summary.csv", index=False)
    (RESULT_DIR / "lymph_node_full_five_method_panel_summary.md").write_text(to_markdown(summary), encoding="utf-8")
    manifest = {
        "dataset": "human_lymph_node_visium_full_subset",
        "source_matrix": "10x Genomics V1_Human_Lymph_Node native Visium full-transcriptome matrix",
        "n_methods": int(summary["method"].nunique()),
        "focus_lr": FOCUS_LR,
        "output": "lymph_node_full_five_method_panel_summary.csv",
        "claim_boundary": (
            "This is a real ST validation/stress-test panel without native CCC ground truth. "
            "Use downstream delta-R2, expected niche flow and negative-control behavior as consistency evidence, not causal proof."
        ),
    }
    (RESULT_DIR / "lymph_node_full_five_method_panel_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


def first(table: pd.DataFrame) -> pd.Series:
    if table.empty:
        return pd.Series(dtype=object)
    return table.iloc[0]


def normalize_method(method: object) -> str:
    return {"spatialself_wavelet_edge_prior": "spatialself"}.get(str(method), str(method))


def value(row: pd.Series, column: str) -> object:
    if row.empty or column not in row:
        return ""
    val = row[column]
    return "" if pd.isna(val) else val


def num(row: pd.Series, column: str) -> float:
    val = value(row, column)
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan


def is_negative_lr(lr_id: object) -> bool:
    return "negative" in str(lr_id).lower()


def flow(direction: pd.DataFrame, method: str, lr_pair: str) -> float:
    sub = direction[direction["method"].eq(method) & direction["lr_pair"].eq(lr_pair)]
    if sub.empty:
        return np.nan
    return float(sub.iloc[0]["expected_flow_score_fraction"])


def ccr7_note(direction: pd.DataFrame, method: str) -> str:
    ccl19 = flow(direction, method, "CCL19-CCR7")
    ccl21 = flow(direction, method, "CCL21-CCR7")
    if np.isfinite(ccl19) and np.isfinite(ccl21) and ccl19 >= 0.8 and ccl21 >= 0.8:
        return "CCL19/CCL21-CCR7 T-zone flow supported"
    if np.isfinite(ccl19) and ccl19 >= 0.8:
        return "CCL19-CCR7 T-zone flow supported"
    if np.isfinite(ccl21) and ccl21 >= 0.8:
        return "CCL21-CCR7 T-zone flow supported"
    return "CCR7 expected-flow support weak or unavailable"


def interpretation(
    method: str,
    imp: pd.Series,
    met: pd.Series,
    direction: pd.DataFrame,
    modules: pd.DataFrame,
) -> str:
    top_negative = is_negative_lr(value(met, "top_lr_by_mean_score"))
    specificity = num(imp, "priority_minus_max_negative")
    ccr7_supported = ccr7_note(direction, method).startswith("CCL")
    clean_modules = int(((modules["n_true_lr"].gt(0)) & (modules["n_high_expression_negative"].eq(0))).sum()) if len(modules) else 0
    if method == "spatalk":
        return "Runnable, but SpaTalk produced no informative native scores for this panel; downstream impact is not interpretable."
    if specificity > 0 and not top_negative:
        return "Good negative-control separation in downstream impact; use as a strong comparator, not a SpatialSelf-only win."
    if specificity > 0:
        return "Downstream negatives are suppressed, but raw mean-score ranking still puts a high-expression negative first."
    if ccr7_supported and clean_modules:
        return "Biological CCR7 flow is supported, but high-expression/receptor-shared negatives remain competitive."
    if top_negative:
        return "High-expression negative controls are competitive; useful as a stress-test comparator."
    return "Mixed support; inspect LR-level outputs before using as a main claim."


def to_markdown(summary: pd.DataFrame) -> str:
    display = summary[
        [
            "method",
            "status",
            "baseline_top_lr_by_mean_score",
            "mean_priority_delta_r2",
            "max_negative_delta_r2",
            "priority_minus_max_negative",
            "ccl19_ccr7_expected_flow_fraction",
            "ccl21_ccr7_expected_flow_fraction",
            "cxcl13_cxcr5_expected_flow_fraction",
            "interpretation",
        ]
    ].copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
    rows = []
    columns = list(display.columns)
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for record in display.astype(str).to_dict(orient="records"):
        rows.append("| " + " | ".join(record[column].replace("|", "/") for column in columns) + " |")
    return "# Native lymph node full five-method panel summary\n\n" + "\n".join(rows) + "\n"


if __name__ == "__main__":
    main()
