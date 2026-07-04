#!/usr/bin/env python
"""Compare expanded-atlas SpatialSelf and COMMOT receiver programs.

This reuses the Phase 12 expanded-atlas LR selection and computes the same
receiver-side whole-transcriptome target-rank scan for SpatialSelf and COMMOT
adapter predictions. It is a COMMOT-style impact proxy, not a full official
COMMOT communication_impact rerun.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ATLAS_DIR = PROJECT_ROOT / "outputs/phase12/atlas_model_expanded"
OUT_DIR = PROJECT_ROOT / "outputs/phase12/expanded_atlas_commot_downstream_comparison"
HELPER_PATH = PROJECT_ROOT / "scripts/phase12/run_phase12_expanded_atlas_downstream_scan.py"

helper_spec = importlib.util.spec_from_file_location("expanded_downstream_helper", HELPER_PATH)
helper = importlib.util.module_from_spec(helper_spec)
assert helper_spec.loader is not None
helper_spec.loader.exec_module(helper)

METHODS = {
    "spatialself_calibrated": {
        "label": "SpatialSelf calibrated",
        "prediction_template": ATLAS_DIR
        / "predictions/{dataset_key}/spatialself_wavelet_edge_prior/{dataset_key}__spatialself_wavelet_predictions.csv",
        "rank_method": "spatialself_wavelet_edge_prior_calibrated",
    },
    "commot_adapter": {
        "label": "COMMOT adapter",
        "prediction_template": ATLAS_DIR / "official_adapter_baselines/commot/{dataset_key}__commot_predictions.csv",
        "rank_method": "commot",
    },
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ranking = pd.read_csv(ATLAS_DIR / "phase12_atlas_model_expanded_calibrated_lr_ranking.csv")
    modules = pd.read_csv(ATLAS_DIR / "phase12_expanded_atlas_modules.csv")
    official_rank = pd.read_csv(ATLAS_DIR / "official_adapter_baselines/phase12_expanded_official_adapter_lr_ranking.csv")

    all_lr_rows: list[pd.DataFrame] = []
    all_dataset_rows: list[dict[str, Any]] = []
    all_status: list[dict[str, Any]] = []
    for dataset_key, spec in helper.DATASETS.items():
        try:
            result = run_dataset(dataset_key, spec, ranking, modules, official_rank)
            all_lr_rows.append(result["lr_summary"])
            all_dataset_rows.extend(result["dataset_summary"])
            all_status.extend(result["status"])
        except Exception as exc:  # noqa: BLE001
            all_status.append({"dataset_key": dataset_key, "method": "all", "status": "failed", "error": repr(exc)})

    lr_summary = pd.concat(all_lr_rows, ignore_index=True) if all_lr_rows else pd.DataFrame()
    comparison = compare_methods(lr_summary)
    dataset_summary = pd.DataFrame(all_dataset_rows)
    status = pd.DataFrame(all_status)

    lr_summary.to_csv(OUT_DIR / "phase12_expanded_commot_downstream_lr_summary.csv", index=False)
    comparison.to_csv(OUT_DIR / "phase12_expanded_commot_downstream_method_comparison.csv", index=False)
    dataset_summary.to_csv(OUT_DIR / "phase12_expanded_commot_downstream_dataset_summary.csv", index=False)
    status.to_csv(OUT_DIR / "phase12_expanded_commot_downstream_status.csv", index=False)
    write_status(dataset_summary, comparison, status)
    write_manifest()
    print(status.to_string(index=False))
    print()
    print(dataset_summary.to_string(index=False))
    print()
    print(comparison.sort_values(["dataset_key", "spatialself_rank"]).head(40).to_string(index=False))


def run_dataset(
    dataset_key: str,
    spec: dict[str, Path],
    ranking: pd.DataFrame,
    modules: pd.DataFrame,
    official_rank: pd.DataFrame,
) -> dict[str, Any]:
    replicate_dir = ATLAS_DIR / "replicates" / dataset_key
    cells = pd.read_csv(replicate_dir / "cells.csv")
    lr_pairs = pd.read_csv(replicate_dir / "lr_pairs.csv")
    candidates = pd.read_csv(
        replicate_dir / "candidate_interactions.csv",
        usecols=["candidate_id", "receiver_id", "lr_id"],
    )
    target_map = helper.load_targets(spec["target_file"])
    dataset_ranking = ranking[ranking["dataset_key"].astype(str).eq(dataset_key)].copy()
    selected = helper.select_lr_pairs(dataset_key, dataset_ranking, modules)
    selected_lr = lr_pairs[(lr_pairs["ligand"] + "-" + lr_pairs["receptor"]).isin(selected["lr_pair"])].copy()
    selected_lr["lr_pair"] = selected_lr["ligand"] + "-" + selected_lr["receptor"]
    selected_lr = selected_lr.merge(selected, on="lr_pair", how="left")
    selected_lr = selected_lr.merge(
        dataset_ranking[
            [
                "lr_id",
                "rank",
                "rank_q95",
                "calibrated_score",
                "calibrated_q95_score",
                "true_lr",
                "false_positive_high_expression",
                "biological_category",
                "mechanism",
            ]
        ],
        on="lr_id",
        how="left",
        suffixes=("", "_rank"),
    ).drop_duplicates("lr_id")
    selected_lr = selected_lr.sort_values(["rank", "lr_pair"])

    counts, genes, barcodes = helper.load_full_tx(spec["full_tx_dir"])
    expression, genes = helper.align_expression_to_cells(counts, genes, barcodes, cells)
    covariates = helper.build_covariates(cells)
    expression.data = np.log1p(expression.data)
    min_cells = max(8, int(round(0.03 * expression.shape[0])))
    expressed = np.asarray((expression > 0).sum(axis=0)).ravel() >= min_cells
    expression = expression[:, expressed].tocsr()
    genes = genes[expressed]
    dense = np.asarray(expression.toarray(), dtype=np.float32)
    residual_expr = helper.residualize_matrix(dense, covariates)
    valid = residual_expr.std(axis=0) > 1e-8
    residual_expr = residual_expr[:, valid]
    genes = genes[valid]
    gene_set = set(genes.tolist())
    ranked_expr = helper.rank_columns(residual_expr)

    lr_rows: list[pd.DataFrame] = []
    dataset_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    for method_key, method_spec in METHODS.items():
        pred_path = Path(str(method_spec["prediction_template"]).format(dataset_key=dataset_key))
        if not pred_path.exists():
            status_rows.append({"dataset_key": dataset_key, "method": method_key, "status": "missing_prediction", "error": str(pred_path)})
            continue
        pred = pd.read_csv(pred_path, usecols=["candidate_id", "score"])
        incoming = helper.receiver_incoming(candidates, pred, cells, selected_lr["lr_id"].astype(str).tolist())
        method_rank = method_rank_table(dataset_key, method_key, dataset_ranking, official_rank)
        summary = scan_method(
            dataset_key,
            method_key,
            str(method_spec["label"]),
            selected_lr,
            incoming,
            covariates,
            ranked_expr,
            genes,
            gene_set,
            target_map,
            method_rank,
        )
        lr_rows.append(summary)
        dataset_rows.append(summarize_dataset_method(dataset_key, method_key, summary))
        status_rows.append(
            {
                "dataset_key": dataset_key,
                "method": method_key,
                "status": "completed",
                "n_cells": int(len(cells)),
                "n_genes_tested": int(len(genes)),
                "n_lr_selected": int(len(selected_lr)),
                "n_lr_scanned": int(len(summary)),
                "error": "",
            }
        )
    return {
        "lr_summary": pd.concat(lr_rows, ignore_index=True) if lr_rows else pd.DataFrame(),
        "dataset_summary": dataset_rows,
        "status": status_rows,
    }


def scan_method(
    dataset_key: str,
    method_key: str,
    method_label: str,
    selected_lr: pd.DataFrame,
    incoming: pd.DataFrame,
    covariates: np.ndarray,
    ranked_expr: np.ndarray,
    genes: np.ndarray,
    gene_set: set[str],
    target_map: dict[str, list[str]],
    method_rank: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for lr in selected_lr.itertuples(index=False):
        lr_id = str(lr.lr_id)
        lr_pair = str(lr.lr_pair)
        if lr_id not in incoming.columns:
            continue
        signal = incoming[lr_id].to_numpy(dtype=float)
        if np.nanstd(signal) <= 1e-10:
            continue
        signal_resid = helper.residualize_vector(signal, covariates)
        if np.nanstd(signal_resid) <= 1e-10:
            continue
        rho = helper.spearman_against_ranked(signal_resid, ranked_expr)
        ranks = helper.rank_descending(rho)
        targets = [gene for gene in target_map.get(lr_pair.upper(), []) if gene in gene_set]
        target_set = set(targets)
        target_rank_values = [int(rank) for gene, rank in zip(genes, ranks) if gene in target_set]
        target_rho = np.asarray([corr for gene, corr in zip(genes, rho) if gene in target_set], dtype=float)
        target_summary = helper.summarize_target_ranks(target_rank_values, len(genes))
        rank_row = method_rank[method_rank["lr_id"].astype(str).eq(lr_id)].head(1)
        rows.append(
            {
                "dataset_key": dataset_key,
                "method": method_key,
                "method_label": method_label,
                "lr_id": lr_id,
                "lr_pair": lr_pair,
                "spatialself_rank": int(lr.rank) if pd.notna(lr.rank) else -1,
                "method_rank": int(rank_row["method_rank"].iloc[0]) if len(rank_row) else -1,
                "method_score": float(rank_row["method_score"].iloc[0]) if len(rank_row) else np.nan,
                "selection_sources": str(lr.selection_sources),
                "true_lr": bool(lr.true_lr),
                "false_positive_high_expression": bool(lr.false_positive_high_expression),
                "biological_category": str(lr.biological_category),
                "mechanism": str(lr.mechanism),
                "n_genes_tested": int(len(genes)),
                "n_targets_present": int(len(target_rank_values)),
                "target_mean_spearman": float(np.nanmean(target_rho)) if len(target_rho) else np.nan,
                "target_mean_r2": float(np.nanmean(np.square(target_rho))) if len(target_rho) else np.nan,
                "all_gene_mean_r2": float(np.nanmean(np.square(rho))),
                "target_delta_r2_vs_all_genes": (
                    float(np.nanmean(np.square(target_rho)) - np.nanmean(np.square(rho))) if len(target_rho) else np.nan
                ),
                **target_summary,
            }
        )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["target_top300_enrichment_p"] = [
            helper.target_enrichment_pvalue(
                int(row.n_genes_tested),
                int(row.n_targets_present),
                min(helper.TOP_GENE_CUTOFF, int(row.n_genes_tested)),
                int(row.n_targets_top300),
            )
            for row in summary.itertuples(index=False)
        ]
        summary["target_top300_enrichment_bh_fdr"] = helper.bh_adjust(
            summary["target_top300_enrichment_p"].to_numpy(dtype=float)
        )
    return summary


def method_rank_table(
    dataset_key: str,
    method_key: str,
    spatialself_rank: pd.DataFrame,
    official_rank: pd.DataFrame,
) -> pd.DataFrame:
    if method_key == "spatialself_calibrated":
        return spatialself_rank.rename(columns={"rank": "method_rank", "calibrated_score": "method_score"})[
            ["lr_id", "method_rank", "method_score"]
        ]
    sub = official_rank[
        official_rank["dataset_key"].astype(str).eq(dataset_key)
        & official_rank["method"].astype(str).eq("commot")
    ].copy()
    if sub.empty:
        return pd.DataFrame(columns=["lr_id", "method_rank", "method_score"])
    return sub.rename(columns={"rank": "method_rank", "score": "method_score"})[["lr_id", "method_rank", "method_score"]]


def summarize_dataset_method(dataset_key: str, method_key: str, summary: pd.DataFrame) -> dict[str, Any]:
    q = pd.to_numeric(summary.get("target_top300_enrichment_bh_fdr", pd.Series(dtype=float)), errors="coerce")
    true = summary[summary["true_lr"].astype(bool)] if not summary.empty else pd.DataFrame()
    neg = summary[summary["false_positive_high_expression"].astype(bool)] if not summary.empty else pd.DataFrame()
    return {
        "dataset_key": dataset_key,
        "method": method_key,
        "n_lr_scanned": int(len(summary)),
        "n_evaluable_target_sets": int(summary["n_targets_present"].fillna(0).astype(int).gt(0).sum()) if not summary.empty else 0,
        "n_target_enriched_q_0_1": int(q.le(0.1).sum()) if len(q) else 0,
        "n_true_lr_target_enriched_q_0_1": int(pd.to_numeric(true.get("target_top300_enrichment_bh_fdr", pd.Series(dtype=float)), errors="coerce").le(0.1).sum()) if len(true) else 0,
        "n_negative_target_enriched_q_0_1": int(pd.to_numeric(neg.get("target_top300_enrichment_bh_fdr", pd.Series(dtype=float)), errors="coerce").le(0.1).sum()) if len(neg) else 0,
        "mean_target_delta_r2_true_lr": float(pd.to_numeric(true.get("target_delta_r2_vs_all_genes", pd.Series(dtype=float)), errors="coerce").mean()) if len(true) else np.nan,
        "mean_target_delta_r2_negative": float(pd.to_numeric(neg.get("target_delta_r2_vs_all_genes", pd.Series(dtype=float)), errors="coerce").mean()) if len(neg) else np.nan,
    }


def compare_methods(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows = []
    for (dataset_key, lr_id), sub in summary.groupby(["dataset_key", "lr_id"], sort=False):
        methods = {row.method: row for row in sub.itertuples(index=False)}
        if "spatialself_calibrated" not in methods or "commot_adapter" not in methods:
            continue
        ss = methods["spatialself_calibrated"]
        cm = methods["commot_adapter"]
        rows.append(
            {
                "dataset_key": dataset_key,
                "lr_id": lr_id,
                "lr_pair": ss.lr_pair,
                "spatialself_rank": int(ss.spatialself_rank),
                "commot_rank": int(cm.method_rank),
                "rank_advantage_vs_commot": safe_float(cm.method_rank) - safe_float(ss.spatialself_rank),
                "true_lr": bool(ss.true_lr),
                "false_positive_high_expression": bool(ss.false_positive_high_expression),
                "selection_sources": ss.selection_sources,
                "n_targets_present": int(ss.n_targets_present),
                "spatialself_target_fdr": safe_float(ss.target_top300_enrichment_bh_fdr),
                "commot_target_fdr": safe_float(cm.target_top300_enrichment_bh_fdr),
                "spatialself_target_enrichment": safe_float(ss.target_top300_enrichment),
                "commot_target_enrichment": safe_float(cm.target_top300_enrichment),
                "spatialself_target_delta_r2": safe_float(ss.target_delta_r2_vs_all_genes),
                "commot_target_delta_r2": safe_float(cm.target_delta_r2_vs_all_genes),
                "spatialself_minus_commot_target_delta_r2": safe_float(ss.target_delta_r2_vs_all_genes)
                - safe_float(cm.target_delta_r2_vs_all_genes),
                "comparison_call": comparison_call(ss, cm),
            }
        )
    return pd.DataFrame(rows)


def comparison_call(ss: object, cm: object) -> str:
    ss_q = safe_float(ss.target_top300_enrichment_bh_fdr)
    cm_q = safe_float(cm.target_top300_enrichment_bh_fdr)
    ss_delta = safe_float(ss.target_delta_r2_vs_all_genes)
    cm_delta = safe_float(cm.target_delta_r2_vs_all_genes)
    ss_supported = np.isfinite(ss_q) and ss_q <= 0.1
    cm_supported = np.isfinite(cm_q) and cm_q <= 0.1
    if ss_supported and not cm_supported:
        return "spatialself_target_support_only"
    if cm_supported and not ss_supported:
        return "commot_target_support_only"
    if ss_supported and cm_supported:
        if ss_delta > cm_delta:
            return "both_supported_spatialself_higher_delta_r2"
        return "both_supported_commot_higher_delta_r2"
    if np.isfinite(ss_delta) and np.isfinite(cm_delta):
        if ss_delta - cm_delta > 0.001:
            return "no_fdr_spatialself_higher_delta_r2"
        if cm_delta - ss_delta > 0.001:
            return "no_fdr_commot_higher_delta_r2"
    return "no_clear_target_support"


def write_status(dataset_summary: pd.DataFrame, comparison: pd.DataFrame, status: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Expanded Atlas COMMOT Downstream Comparison",
        "",
        "## Scope",
        "",
        "This compares SpatialSelf calibrated and COMMOT adapter scores on the same expanded-atlas top/anchor/module LR selection using receiver-side whole-transcriptome target-rank scans. It is a COMMOT-style impact proxy, not a full official COMMOT communication_impact rerun.",
        "",
        "## Dataset Summary",
        "",
    ]
    for row in dataset_summary.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.method}`: LR={int(row.n_lr_scanned)}, evaluable targets={int(row.n_evaluable_target_sets)}, target-enriched q<=0.1={int(row.n_target_enriched_q_0_1)}, true target-enriched={int(row.n_true_lr_target_enriched_q_0_1)}, negative target-enriched={int(row.n_negative_target_enriched_q_0_1)}"
        )
    if not comparison.empty:
        lines += ["", "## Method Comparison Highlights", ""]
        for dataset_key, sub in comparison.groupby("dataset_key"):
            calls = sub["comparison_call"].value_counts()
            lines.append(
                f"- `{dataset_key}`: "
                + "; ".join(f"{idx}={int(val)}" for idx, val in calls.items())
            )
    lines += [
        "",
        "## Claim Boundary",
        "",
        "Use this table as same-universe downstream/impact comparison evidence. Do not describe it as a fully tuned official COMMOT communication_impact benchmark.",
    ]
    (OUT_DIR / "phase12_expanded_commot_downstream_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest() -> None:
    manifest = {
        "task": "phase12_expanded_atlas_commot_downstream_comparison",
        "outputs": [
            rel(OUT_DIR / "phase12_expanded_commot_downstream_lr_summary.csv"),
            rel(OUT_DIR / "phase12_expanded_commot_downstream_method_comparison.csv"),
            rel(OUT_DIR / "phase12_expanded_commot_downstream_dataset_summary.csv"),
            rel(OUT_DIR / "phase12_expanded_commot_downstream_status.csv"),
            rel(OUT_DIR / "phase12_expanded_commot_downstream_status.md"),
        ],
        "claim_boundary": "COMMOT-style adapter comparison, not full official COMMOT communication_impact.",
    }
    (OUT_DIR / "phase12_expanded_commot_downstream_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def safe_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
