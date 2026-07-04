#!/usr/bin/env python
"""Summarize Phase 5 real-background validation for manuscript drafting."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_LABELS = {
    "lymph_node_real_background_smoke": "lymph_node",
    "tonsil_real_background_smoke": "tonsil",
    "melanoma_real_background_smoke": "melanoma",
}

METHOD_LABELS = {
    "spatialself_wavelet_edge_prior": "SpatialSelf",
    "spatialself_wavelet": "SpatialSelf base",
    "commot": "COMMOT",
    "cellchat": "CellChat",
    "spatalk": "SpaTalk",
    "cellnest": "CellNEST",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-summary", default="outputs/phase2_revised/goal1_full_baseline_matched_validation_all/goal1_baseline_matched_method_summary.csv")
    parser.add_argument("--lr-summary", default="outputs/phase2_revised/goal1_full_baseline_matched_validation_all/goal1_baseline_matched_lr_control_summary.csv")
    parser.add_argument("--claims", default="outputs/phase2_revised/goal1_full_baseline_matched_validation_all/goal1_baseline_matched_claims.csv")
    parser.add_argument("--output-dir", default="outputs/phase5/manuscript_submission_lock/goal2_real_validation")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    method_rows = _read_csv(PROJECT_ROOT / args.method_summary)
    lr_rows = _read_csv(PROJECT_ROOT / args.lr_summary)
    claim_rows = _read_csv(PROJECT_ROOT / args.claims)

    method_summary = _build_method_summary(method_rows)
    tissue_summary = _build_tissue_summary(method_summary, claim_rows)
    lr_panel = _build_lr_panel(lr_rows)
    claim_table = _build_claims(tissue_summary, method_summary)

    _write_csv(output_dir / "phase5_real_validation_method_summary.csv", method_summary, method_summary[0].keys())
    _write_csv(output_dir / "phase5_real_validation_tissue_summary.csv", tissue_summary, tissue_summary[0].keys())
    _write_csv(output_dir / "phase5_real_validation_lr_panel.csv", lr_panel, lr_panel[0].keys())
    _write_csv(output_dir / "phase5_real_validation_claims.csv", claim_table, claim_table[0].keys())
    manifest = {
        "method_summary": _rel(output_dir / "phase5_real_validation_method_summary.csv"),
        "tissue_summary": _rel(output_dir / "phase5_real_validation_tissue_summary.csv"),
        "lr_panel": _rel(output_dir / "phase5_real_validation_lr_panel.csv"),
        "claims": _rel(output_dir / "phase5_real_validation_claims.csv"),
        "source_method_summary": args.method_summary,
        "source_lr_summary": args.lr_summary,
        "source_claims": args.claims,
        "claim_level": "real_background_semisynthetic_and_proxy_validation_not_native_ground_truth",
    }
    (output_dir / "phase5_real_validation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def _build_method_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        dataset = DATASET_LABELS.get(row.get("dataset", ""), row.get("dataset", ""))
        method = row.get("method", "")
        pos_rank = _float(row.get("mean_positive_rank"))
        neg_rank = _float(row.get("mean_negative_rank"))
        pos_spear = _float(row.get("mean_positive_receiver_response_proxy_spearman"))
        neg_spear = _float(row.get("mean_negative_receiver_response_proxy_spearman"))
        out.append(
            {
                "dataset": dataset,
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "n_positive_controls": _int(row.get("n_positive_controls")),
                "n_negative_controls": _int(row.get("n_negative_controls")),
                "mean_positive_rank": _fmt(pos_rank),
                "mean_negative_rank": _fmt(neg_rank),
                "rank_separation_negative_minus_positive": _fmt(neg_rank - pos_rank),
                "positive_top3_count": _int(row.get("positive_top3_count")),
                "negative_top3_count": _int(row.get("negative_top3_count")),
                "rank_validation_pass": row.get("rank_validation_pass", ""),
                "mean_positive_receiver_response_proxy_spearman": _fmt(pos_spear),
                "mean_negative_receiver_response_proxy_spearman": _fmt(neg_spear),
                "receiver_response_proxy_delta": _fmt(pos_spear - neg_spear),
                "empirical_fdr_at_q95": _fmt(_float(row.get("empirical_fdr_at_q95"))),
                "high_expression_false_positive_rate_at_q95": _fmt(_float(row.get("high_expression_false_positive_rate_at_q95"))),
                "receiver_response_proxy_level": row.get("receiver_response_proxy_level", ""),
            }
        )
    return out


def _build_tissue_summary(method_summary: list[dict[str, Any]], claim_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    claims_by_dataset = {
        DATASET_LABELS.get(row.get("dataset", ""), row.get("dataset", "")): row
        for row in claim_rows
    }
    out: list[dict[str, Any]] = []
    for dataset in sorted({row["dataset"] for row in method_summary}):
        rows = [row for row in method_summary if row["dataset"] == dataset]
        spatial = _first([row for row in rows if row["method"] == "spatialself_wavelet_edge_prior"])
        baselines = [row for row in rows if row["method"] not in {"spatialself_wavelet_edge_prior", "spatialself_wavelet"}]
        best_fp = min((_float(row["high_expression_false_positive_rate_at_q95"]) for row in rows), default=float("nan"))
        spatial_fp = _float(spatial.get("high_expression_false_positive_rate_at_q95")) if spatial else float("nan")
        baseline_min_fp = min((_float(row["high_expression_false_positive_rate_at_q95"]) for row in baselines), default=float("nan"))
        claim = claims_by_dataset.get(dataset, {})
        out.append(
            {
                "dataset": dataset,
                "n_methods": len(rows),
                "spatialself_mean_positive_rank": spatial.get("mean_positive_rank", "") if spatial else "",
                "spatialself_mean_negative_rank": spatial.get("mean_negative_rank", "") if spatial else "",
                "spatialself_rank_separation": spatial.get("rank_separation_negative_minus_positive", "") if spatial else "",
                "spatialself_positive_top3_count": spatial.get("positive_top3_count", "") if spatial else "",
                "spatialself_negative_top3_count": spatial.get("negative_top3_count", "") if spatial else "",
                "spatialself_rank_validation_pass": spatial.get("rank_validation_pass", "") if spatial else "",
                "spatialself_proxy_spearman_delta": spatial.get("receiver_response_proxy_delta", "") if spatial else "",
                "spatialself_empirical_fdr_q95": spatial.get("empirical_fdr_at_q95", "") if spatial else "",
                "spatialself_high_expression_fp_rate_q95": _fmt(spatial_fp),
                "best_method_by_rank": claim.get("best_rank_method", ""),
                "lowest_fp_method": claim.get("lowest_high_expression_fp_method", ""),
                "spatialself_lowest_fp": claim.get("spatialself_lowest_fp", ""),
                "best_fp_rate_any_method": _fmt(best_fp),
                "baseline_min_high_expression_fp_rate_q95": _fmt(baseline_min_fp),
                "spatialself_vs_baseline_min_fp_delta": _fmt(spatial_fp - baseline_min_fp),
                "claim_level": claim.get("claim_level", "real_background_semisynthetic_proxy"),
            }
        )
    return out


def _build_lr_panel(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("method") != "spatialself_wavelet_edge_prior":
            continue
        dataset = DATASET_LABELS.get(row.get("dataset", ""), row.get("dataset", ""))
        out.append(
            {
                "dataset": dataset,
                "lr_id": row.get("lr_id", ""),
                "ligand": row.get("ligand", ""),
                "receptor": row.get("receptor", ""),
                "mechanism": row.get("mechanism", ""),
                "positive_control": row.get("positive_control", ""),
                "negative_control": row.get("negative_control", ""),
                "weak_signal": row.get("weak_signal", ""),
                "false_positive_high_expression": row.get("false_positive_high_expression", ""),
                "global_rank": row.get("global_rank", ""),
                "mean_score": _fmt(_float(row.get("mean_score"))),
                "q95_score": _fmt(_float(row.get("q95_score"))),
                "receiver_response_proxy_spearman": _fmt(_float(row.get("receiver_response_proxy_spearman"))),
                "receiver_response_proxy_status": row.get("receiver_response_proxy_status", ""),
            }
        )
    return out


def _build_claims(tissue_summary: list[dict[str, Any]], method_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spatial = [row for row in method_summary if row["method"] == "spatialself_wavelet_edge_prior"]
    n_tissues = len(tissue_summary)
    n_rank_pass = sum(1 for row in tissue_summary if _truthy(row.get("spatialself_rank_validation_pass")))
    n_lowest_fp = sum(1 for row in tissue_summary if _truthy(row.get("spatialself_lowest_fp")))
    fp_rates = [_float(row.get("spatialself_high_expression_fp_rate_q95")) for row in tissue_summary]
    deltas = [_float(row.get("spatialself_proxy_spearman_delta")) for row in tissue_summary]
    return [
        {
            "claim_id": "P5_RV1_three_tissue_control_ranking",
            "status": "supported" if n_rank_pass == n_tissues and n_tissues >= 3 else "partial",
            "allowed_wording": "Across lymph node, tonsil, and melanoma real-background panels, SpatialSelf ranks injected positive controls above high-expression negative controls.",
            "disallowed_wording": "This proves native real-tissue ground truth performance.",
            "evidence_value": f"rank_pass={n_rank_pass}/{n_tissues}",
            "limitation": "Semi-synthetic real-background or prior-control validation, not native complete ground truth.",
        },
        {
            "claim_id": "P5_RV2_high_expression_fp_suppression",
            "status": "supported" if n_lowest_fp == n_tissues and n_tissues >= 3 else "partial",
            "allowed_wording": "SpatialSelf keeps high-expression false-positive rates low in all three real-background panels, with SpaTalk noted separately where conservative behavior appears.",
            "disallowed_wording": "SpatialSelf always has the lowest false-positive rate in every real dataset and every setting.",
            "evidence_value": f"lowest_fp={n_lowest_fp}/{n_tissues};spatialself_fp_range={_fmt(min(fp_rates))}-{_fmt(max(fp_rates))}",
            "limitation": "False-positive labels are injected controls; native random/shuffled controls should be added for stronger real-data claims.",
        },
        {
            "claim_id": "P5_RV3_receiver_response_proxy",
            "status": "supported" if all(delta > 0 for delta in deltas) else "partial",
            "allowed_wording": "SpatialSelf positive controls show higher receptor-expression proxy correlation than negative controls across the three panels.",
            "disallowed_wording": "This is curated downstream validation across all tissues.",
            "evidence_value": "proxy_delta_by_tissue=" + "|".join(f"{row['dataset']}:{row['spatialself_proxy_spearman_delta']}" for row in tissue_summary),
            "limitation": "Tonsil and melanoma use receiver-response proxy rather than curated downstream target validation.",
        },
        {
            "claim_id": "P5_RV4_baseline_matched_scope",
            "status": "supported" if all(int(row["n_methods"]) >= 6 for row in tissue_summary) else "partial",
            "allowed_wording": "The control-ranking analysis is baseline-matched against COMMOT, CellChat, SpaTalk, CellNEST, and SpatialSelf variants.",
            "disallowed_wording": "All external methods are fully optimized final submissions.",
            "evidence_value": "methods_per_tissue=" + "|".join(f"{row['dataset']}:{row['n_methods']}" for row in tissue_summary),
            "limitation": "CellNEST and adapter-based methods retain implementation/status caveats from Phase 4.",
        },
    ]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: Any) -> None:
    columns = list(columns)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _first(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return rows[0] if rows else {}


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "1.0", "yes", "y", "supported"}


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _fmt(value: float) -> str:
    if value != value:
        return ""
    return f"{value:.6g}"


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
