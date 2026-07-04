#!/usr/bin/env python
"""Summarize Phase 4 downstream target validation evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        default=(
            "outputs/phase4/manuscript_evidence_lock/goal5_downstream_key_lymph_node/"
            "human_lymph_node_visium_replicate_00__spatialself_wavelet_predictions__downstream_target_correlation.csv"
        ),
    )
    parser.add_argument("--target-panel", default="configs/phase4_curated_lr_targets.csv")
    parser.add_argument("--output-dir", default="outputs/phase4/manuscript_evidence_lock/goal5_downstream_key_lymph_node")
    args = parser.parse_args()

    result_path = PROJECT_ROOT / args.result
    target_path = PROJECT_ROOT / args.target_panel
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results = _read_csv(result_path)
    targets = _read_csv(target_path)
    negative_pairs = {
        f"{row.get('ligand')}::{row.get('receptor')}"
        for row in targets
        if row.get("evidence_scope") == "phase4_negative_control_response"
    }
    rows = []
    for row in results:
        key = f"{row.get('ligand')}::{row.get('receptor')}"
        is_negative = key in negative_pairs or "false_positive" in row.get("mechanism", "")
        rows.append(
            {
                **row,
                "control_class": "negative_high_expression" if is_negative else "positive_control",
                "phase4_target_panel": _rel(target_path),
            }
        )

    positives = [row for row in rows if row["control_class"] == "positive_control" and row.get("status") == "tested"]
    negatives = [row for row in rows if row["control_class"] == "negative_high_expression" and row.get("status") == "tested"]
    pos_mean = _mean(_float(row.get("spearman_r")) for row in positives)
    neg_mean = _mean(_float(row.get("spearman_r")) for row in negatives)
    min_pos_targets = min([int(float(row.get("n_targets_present") or 0)) for row in positives] or [0])
    min_neg_targets = min([int(float(row.get("n_targets_present") or 0)) for row in negatives] or [0])
    all_positive_tested = len(positives) >= 3 and min_pos_targets >= 3
    all_negative_tested = len(negatives) >= 2 and min_neg_targets >= 3
    delta = pos_mean - neg_mean

    summary = [
        {
            "dataset": "real_background_v2_replicated_lymph_node",
            "method": "spatialself_wavelet_edge_prior",
            "n_positive_tested": len(positives),
            "n_negative_tested": len(negatives),
            "min_positive_targets_present": min_pos_targets,
            "min_negative_targets_present": min_neg_targets,
            "mean_positive_spearman": pos_mean,
            "mean_negative_spearman": neg_mean,
            "positive_negative_spearman_delta": delta,
            "all_positive_controls_tested": all_positive_tested,
            "all_negative_controls_tested": all_negative_tested,
            "coverage_limited": not (all_positive_tested and all_negative_tested),
            "claim_level": "key_lymph_node_downstream_supported_not_global_downstream_superiority",
            "interpretation": (
                "All key lymph-node positive and high-expression negative controls have receiver target signatures. "
                "Positive controls show higher mean target correlation, but negative high-expression controls remain correlated, "
                "so this supports a cautious key-panel downstream validation rather than global downstream specificity."
            ),
        }
    ]
    claims = [
        {
            "claim_id": "P4_D1_key_lymph_node_downstream_coverage",
            "status": "supported" if all_positive_tested and all_negative_tested else "partial",
            "allowed_wording": "A curated key lymph-node downstream panel tests all injected positive and high-expression negative LR controls.",
            "disallowed_wording": "Downstream validation is globally complete across all tissues.",
            "evidence": "phase4_downstream_key_lymph_node_summary.csv",
        },
        {
            "claim_id": "P4_D2_positive_negative_delta",
            "status": "supported" if delta > 0 else "unsupported",
            "allowed_wording": "In the key lymph-node panel, positive controls have higher mean receiver target correlation than high-expression negative controls.",
            "disallowed_wording": "Downstream signatures perfectly separate positives from high-expression negatives.",
            "evidence": "phase4_downstream_key_lymph_node_summary.csv",
        },
        {
            "claim_id": "P4_D3_cross_tissue_downstream",
            "status": "partial",
            "allowed_wording": "Cross-tissue downstream validation requires regenerating real-background datasets with expanded required target genes.",
            "disallowed_wording": "The current downstream panel proves global biological validation across lymph node, tonsil, and melanoma.",
            "evidence": "phase4_downstream_key_lymph_node_summary.csv",
        },
    ]

    _write_csv(output_dir / "phase4_downstream_key_lymph_node_results.csv", rows, rows[0].keys() if rows else [])
    _write_csv(output_dir / "phase4_downstream_key_lymph_node_summary.csv", summary, summary[0].keys())
    _write_csv(output_dir / "phase4_downstream_key_lymph_node_claims.csv", claims, claims[0].keys())
    manifest = {
        "analysis_label": "phase4_key_lymph_node_downstream_validation",
        "result": _rel(result_path),
        "target_panel": _rel(target_path),
        "outputs": {
            "results": _rel(output_dir / "phase4_downstream_key_lymph_node_results.csv"),
            "summary": _rel(output_dir / "phase4_downstream_key_lymph_node_summary.csv"),
            "claims": _rel(output_dir / "phase4_downstream_key_lymph_node_claims.csv"),
        },
    }
    (output_dir / "phase4_downstream_key_lymph_node_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary[0], indent=2))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: Any) -> None:
    columns = list(columns)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _mean(values: Any) -> float:
    vals = [value for value in values if value == value]
    return sum(vals) / len(vals) if vals else float("nan")


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
