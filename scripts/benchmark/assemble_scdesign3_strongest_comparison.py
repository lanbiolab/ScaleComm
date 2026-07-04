#!/usr/bin/env python
"""Assemble a clean comparison table for the scDesign3 strongest benchmark panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METHOD_ORDER = {
    "spatialself_range_context": 0,
    "commot": 1,
    "cellchat": 2,
    "spatalk": 3,
    "cellnest": 4,
    "deeptalk": 5,
}

METHOD_LABEL = {
    "spatialself_range_context": "SpatialSelf",
    "commot": "COMMOT",
    "cellchat": "CellChat",
    "spatalk": "SpaTalk",
    "cellnest": "CellNEST",
    "deeptalk": "DeepTalk",
}


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _metric_row(dataset: str, method: str, metric_path: Path, *, status: str = "completed", note: str = "") -> dict:
    metrics = _read_json(metric_path)
    return {
        "dataset": dataset,
        "replicate": metrics.get("replicate", ""),
        "method": method,
        "method_label": METHOD_LABEL.get(method, method),
        "status": metrics.get("status", status) or status,
        "note": note or metrics.get("spatalk_status", "") or "",
        "auroc": metrics.get("auroc"),
        "aupr": metrics.get("aupr"),
        "top_k_precision": metrics.get("top_k_precision"),
        "weak_signal_recall": metrics.get("weak_signal_recall_at_threshold"),
        "empirical_fdr": metrics.get("empirical_fdr_at_threshold"),
        "n_candidates": metrics.get("n_candidates"),
        "n_projected_candidates": metrics.get("n_projected_candidates"),
        "metrics_path": str(metric_path),
    }


def _add_dataset_ranks(table: pd.DataFrame) -> pd.DataFrame:
    ranked = table.copy()
    ranked["aupr_rank"] = (
        ranked.groupby("dataset")["aupr"]
        .rank(ascending=False, method="min", na_option="bottom")
        .astype("Int64")
    )
    ranked["fdr_rank"] = (
        ranked.groupby("dataset")["empirical_fdr"]
        .rank(ascending=True, method="min", na_option="bottom")
        .astype("Int64")
    )
    return ranked


def build_tables(
    *,
    lymph_root: Path,
    melanoma_root: Path,
    lymph_deeptalk_root: Path,
    lymph_deeptalk_official_root: Path | None,
    melanoma_deeptalk_official_root: Path | None,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    # Lymph
    rows.append(
        _metric_row(
            "lymph_scdesign3",
            "spatialself_range_context",
            lymph_root / "spatialself_range_context/reference_lymph_node_scdesign3_replicate_00__spatialself_range_context_metrics.json",
        )
    )
    rows.append(
        _metric_row(
            "lymph_scdesign3",
            "commot",
            lymph_root / "commot/reference_lymph_node_scdesign3_replicate_00__commot_metrics.json",
        )
    )
    rows.append(
        _metric_row(
            "lymph_scdesign3",
            "cellchat",
            lymph_root / "cellchat/reference_lymph_node_scdesign3_replicate_00__cellchat_metrics.json",
        )
    )
    rows.append(
        _metric_row(
            "lymph_scdesign3",
            "spatalk",
            lymph_root / "spatalk/reference_lymph_node_scdesign3_replicate_00__spatalk_metrics.json",
        )
    )
    rows.append(
        {
            "dataset": "lymph_scdesign3",
            "replicate": "reference_lymph_node_scdesign3_replicate_00",
            "method": "cellnest",
            "method_label": "CellNEST",
            "status": "failed_zero_edge_graph",
            "note": "CellNEST training graph had 0 edges; crashed in remove_self_loops IndexError.",
            "auroc": None,
            "aupr": None,
            "top_k_precision": None,
            "weak_signal_recall": None,
            "empirical_fdr": None,
            "n_candidates": 10596,
            "n_projected_candidates": 0,
            "metrics_path": str(lymph_root / "cellnest/logs/reference_lymph_node_scdesign3_replicate_00__train.log"),
        }
    )
    lymph_deeptalk_metric = (
        lymph_deeptalk_official_root / "reference_lymph_node_scdesign3_replicate_00__deeptalk_metrics.json"
        if lymph_deeptalk_official_root is not None
        else lymph_deeptalk_root / "reference_lymph_node_scdesign3_replicate_00__deeptalk_metrics.json"
    )
    rows.append(
        _metric_row(
            "lymph_scdesign3",
            "deeptalk",
            lymph_deeptalk_metric,
            status="completed_from_existing_predict_ccc" if lymph_deeptalk_official_root is not None else "model_required",
            note="Official DeepTalk prediction completed with cross-LR pretrained model." if lymph_deeptalk_official_root is not None else "DeepTalk inputs prepared, but no predict_ccc output was available to project.",
        )
    )

    # Melanoma
    rows.append(
        _metric_row(
            "melanoma_scdesign3",
            "spatialself_range_context",
            melanoma_root / "spatialself_range_context/reference_melanoma_scdesign3_replicate_00__spatialself_range_context_metrics.json",
        )
    )
    rows.append(
        _metric_row(
            "melanoma_scdesign3",
            "commot",
            melanoma_root / "commot/reference_melanoma_scdesign3_replicate_00__commot_metrics.json",
        )
    )
    rows.append(
        _metric_row(
            "melanoma_scdesign3",
            "cellchat",
            melanoma_root / "cellchat/reference_melanoma_scdesign3_replicate_00__cellchat_metrics.json",
        )
    )
    rows.append(
        _metric_row(
            "melanoma_scdesign3",
            "spatalk",
            melanoma_root / "spatalk/reference_melanoma_scdesign3_replicate_00__spatalk_metrics.json",
        )
    )
    rows.append(
        _metric_row(
            "melanoma_scdesign3",
            "cellnest",
            melanoma_root / "cellnest/reference_melanoma_scdesign3_replicate_00__cellnest_metrics.json",
        )
    )
    melanoma_deeptalk_metric = (
        melanoma_deeptalk_official_root / "reference_melanoma_scdesign3_replicate_00__deeptalk_metrics.json"
        if melanoma_deeptalk_official_root is not None
        else melanoma_root / "deeptalk/reference_melanoma_scdesign3_replicate_00__deeptalk_metrics.json"
    )
    rows.append(
        _metric_row(
            "melanoma_scdesign3",
            "deeptalk",
            melanoma_deeptalk_metric,
            status="completed_from_existing_predict_ccc" if melanoma_deeptalk_official_root is not None else "model_required",
            note="Official DeepTalk prediction completed with cross-LR pretrained model." if melanoma_deeptalk_official_root is not None else "DeepTalk inputs prepared, but no predict_ccc output was available to project.",
        )
    )

    detail = pd.DataFrame(rows)
    detail["_method_order"] = detail["method"].map(METHOD_ORDER).fillna(999)
    detail = detail.sort_values(["dataset", "_method_order", "method"]).drop(columns="_method_order")
    detail = _add_dataset_ranks(detail)

    scored = detail[detail["status"].astype(str).str.startswith("completed")].copy()
    summary = (
        scored.groupby(["method", "method_label"], as_index=False)[
            ["auroc", "aupr", "top_k_precision", "weak_signal_recall", "empirical_fdr"]
        ]
        .mean(numeric_only=True)
    )
    summary["n_datasets_scored"] = scored.groupby("method")["dataset"].nunique().reindex(summary["method"]).to_list()
    summary["mean_aupr_rank"] = summary["aupr"].rank(ascending=False, method="min").astype("Int64")
    summary["mean_fdr_rank"] = summary["empirical_fdr"].rank(ascending=True, method="min").astype("Int64")
    summary["_method_order"] = summary["method"].map(METHOD_ORDER).fillna(999)
    summary = summary.sort_values(["_method_order", "method"]).drop(columns="_method_order")

    detail_path = output_dir / "scdesign3_strongest_comparison_detail.csv"
    summary_path = output_dir / "scdesign3_strongest_comparison_summary.csv"
    markdown_path = output_dir / "scdesign3_strongest_comparison_table.md"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    markdown_path.write_text(_to_markdown(detail), encoding="utf-8")
    return detail, summary


def _to_markdown(detail: pd.DataFrame) -> str:
    table = detail[
        [
            "dataset",
            "method_label",
            "status",
            "aupr",
            "top_k_precision",
            "weak_signal_recall",
            "empirical_fdr",
            "aupr_rank",
            "fdr_rank",
            "note",
        ]
    ].copy()
    for column in ("aupr", "top_k_precision", "weak_signal_recall", "empirical_fdr"):
        table[column] = table[column].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    headers = list(table.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in table.itertuples(index=False):
        values = ["" if pd.isna(value) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lymph-root",
        default="outputs/reference_based_benchmark/full_scdesign3_strongest/lymph",
    )
    parser.add_argument(
        "--melanoma-root",
        default="outputs/reference_based_benchmark/full_scdesign3_strongest/melanoma",
    )
    parser.add_argument(
        "--lymph-deeptalk-root",
        default="outputs/reference_based_benchmark/full_scdesign3_strongest_lymph_deeptalk_only/deeptalk",
    )
    parser.add_argument(
        "--lymph-deeptalk-official-root",
        default="outputs/reference_based_benchmark/deeptalk_official_scdesign3/lymph",
    )
    parser.add_argument(
        "--melanoma-deeptalk-official-root",
        default="outputs/reference_based_benchmark/deeptalk_official_scdesign3/melanoma",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/reference_based_benchmark/full_scdesign3_strongest/final_tables",
    )
    args = parser.parse_args()

    detail, summary = build_tables(
        lymph_root=Path(args.lymph_root),
        melanoma_root=Path(args.melanoma_root),
        lymph_deeptalk_root=Path(args.lymph_deeptalk_root),
        lymph_deeptalk_official_root=Path(args.lymph_deeptalk_official_root) if args.lymph_deeptalk_official_root else None,
        melanoma_deeptalk_official_root=Path(args.melanoma_deeptalk_official_root) if args.melanoma_deeptalk_official_root else None,
        output_dir=Path(args.output_dir),
    )
    print(summary.to_string(index=False))
    print(f"\nWrote detailed comparison to {Path(args.output_dir) / 'scdesign3_strongest_comparison_detail.csv'}")


if __name__ == "__main__":
    main()
