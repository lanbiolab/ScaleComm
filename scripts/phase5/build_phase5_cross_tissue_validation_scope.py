#!/usr/bin/env python
"""Build Phase 5 cross-tissue validation scope tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASETS = ["lymph_node", "tonsil", "melanoma"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/phase5/manuscript_submission_lock/goal2_cross_tissue_scope")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tissue_summary = _read_csv(PROJECT_ROOT / "outputs/phase5/manuscript_submission_lock/goal2_real_validation/phase5_real_validation_tissue_summary.csv")
    bio_index = _read_csv(PROJECT_ROOT / "outputs/phase3/end_to_end/phase3_biological_validation_index.csv")
    key_downstream = _read_csv(PROJECT_ROOT / "outputs/phase4/manuscript_evidence_lock/goal5_downstream_key_lymph_node/phase4_downstream_key_lymph_node_summary.csv")
    fdr_rows = _collect_fdr_scope()

    scope_rows = _build_scope(tissue_summary, bio_index, key_downstream, fdr_rows)
    claims = _build_claims(scope_rows)
    _write_csv(output_dir / "phase5_cross_tissue_validation_scope.csv", scope_rows, scope_rows[0].keys())
    _write_csv(output_dir / "phase5_cross_tissue_validation_claims.csv", claims, claims[0].keys())
    manifest = {
        "scope": _rel(output_dir / "phase5_cross_tissue_validation_scope.csv"),
        "claims": _rel(output_dir / "phase5_cross_tissue_validation_claims.csv"),
        "claim_level": "cross_tissue_validation_scope_guardrail",
    }
    (output_dir / "phase5_cross_tissue_validation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def _build_scope(
    tissue_summary: list[dict[str, str]],
    bio_index: list[dict[str, str]],
    key_downstream: list[dict[str, str]],
    fdr_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tissue_by_dataset = {row.get("dataset"): row for row in tissue_summary}
    bio_by_dataset = _group_bio(bio_index)
    fdr_by_dataset = {row["dataset"]: row for row in fdr_rows}
    key = key_downstream[0] if key_downstream else {}
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        tissue = tissue_by_dataset.get(dataset, {})
        bio = bio_by_dataset.get(dataset, {})
        fdr = fdr_by_dataset.get(dataset, {})
        curated_key = dataset == "lymph_node" and key and not _truthy(key.get("coverage_limited"))
        proxy_supported = _float(tissue.get("spatialself_proxy_spearman_delta")) > 0
        control_supported = _truthy(tissue.get("spatialself_rank_validation_pass"))
        fdr_available = _int(fdr.get("n_group_summary_rows")) > 0
        if curated_key:
            downstream_scope = "key_curated_downstream_supported"
        elif proxy_supported:
            downstream_scope = "receiver_response_proxy_supported_not_curated_global"
        else:
            downstream_scope = "downstream_not_supported"
        allowed = _allowed_wording(dataset, curated_key, proxy_supported, control_supported, fdr_available)
        rows.append(
            {
                "dataset": dataset,
                "control_ranking_status": "supported" if control_supported else "partial",
                "positive_top3_count": tissue.get("spatialself_positive_top3_count", ""),
                "negative_top3_count": tissue.get("spatialself_negative_top3_count", ""),
                "high_expression_fp_rate_q95": tissue.get("spatialself_high_expression_fp_rate_q95", ""),
                "downstream_scope": downstream_scope,
                "receiver_response_proxy_delta": tissue.get("spatialself_proxy_spearman_delta", ""),
                "curated_downstream_coverage": "complete_key_panel" if curated_key else "not_key_curated_complete",
                "fdr_scope": "available_group_summary" if fdr_available else "not_available",
                "fdr_group_summary_rows": fdr.get("n_group_summary_rows", 0),
                "fdr_max_permutations": fdr.get("max_permutations", ""),
                "biological_index_contexts": bio.get("contexts", ""),
                "biological_index_limitations": bio.get("limitations", ""),
                "allowed_wording": allowed,
                "disallowed_wording": "Do not claim complete native real-tissue ground truth or global cross-tissue curated downstream validation.",
                "evidence_tier": _evidence_tier(curated_key, proxy_supported, control_supported, fdr_available),
            }
        )
    return rows


def _collect_fdr_scope() -> list[dict[str, Any]]:
    candidates = [
        ("lymph_node", PROJECT_ROOT / "outputs/phase4/manuscript_evidence_lock/goal3_fdr_500perm_lymph_node/goal5_formal_fdr_group_summary.csv"),
        ("lymph_node", PROJECT_ROOT / "outputs/phase2_revised/goal5_real_background_fdr_panel/real_background_v2_formal_lymph_node/goal5_formal_fdr_group_summary.csv"),
        ("tonsil", PROJECT_ROOT / "outputs/phase2_revised/goal5_real_background_fdr_panel/real_background_v2_formal_tonsil/goal5_formal_fdr_group_summary.csv"),
        ("melanoma", PROJECT_ROOT / "outputs/phase2_revised/goal5_real_background_fdr_panel/real_background_v2_formal_melanoma/goal5_formal_fdr_group_summary.csv"),
    ]
    grouped: dict[str, list[dict[str, str]]] = {}
    for dataset, path in candidates:
        rows = _read_csv(path)
        for row in rows:
            row = dict(row)
            row["_source"] = _rel(path)
            grouped.setdefault(dataset, []).append(row)
    out: list[dict[str, Any]] = []
    for dataset, rows in grouped.items():
        out.append(
            {
                "dataset": dataset,
                "n_group_summary_rows": len(rows),
                "max_permutations": max((_int(row.get("n_permutations")) for row in rows), default=0),
                "sources": "|".join(sorted({row.get("_source", "") for row in rows})),
            }
        )
    return out


def _group_bio(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    alias = {
        "real_background_lymph_node": "lymph_node",
        "real_background_tonsil": "tonsil",
        "high_resolution_tonsil_subset": "tonsil",
        "real_background_melanoma": "melanoma",
    }
    grouped: dict[str, dict[str, set[str]]] = {}
    for row in rows:
        dataset = alias.get(row.get("dataset_id", ""), row.get("dataset_id", ""))
        if dataset not in DATASETS:
            continue
        entry = grouped.setdefault(dataset, {"contexts": set(), "limitations": set()})
        if row.get("validation_context"):
            entry["contexts"].add(row["validation_context"])
        if row.get("limitations"):
            entry["limitations"].add(row["limitations"])
    return {
        dataset: {
            "contexts": "|".join(sorted(values["contexts"])),
            "limitations": "|".join(sorted(values["limitations"])),
        }
        for dataset, values in grouped.items()
    }


def _allowed_wording(dataset: str, curated_key: bool, proxy_supported: bool, control_supported: bool, fdr_available: bool) -> str:
    name = {"lymph_node": "lymph node", "tonsil": "tonsil", "melanoma": "melanoma"}[dataset]
    parts = []
    if control_supported:
        parts.append(f"{name} shows positive/negative control separation")
    if curated_key:
        parts.append("key curated downstream validation is available")
    elif proxy_supported:
        parts.append("receiver-response proxy is positive")
    if fdr_available:
        parts.append("group-level FDR outputs are available")
    return "; ".join(parts) + "."


def _evidence_tier(curated_key: bool, proxy_supported: bool, control_supported: bool, fdr_available: bool) -> str:
    if curated_key and control_supported and fdr_available:
        return "key_curated_validation"
    if proxy_supported and control_supported and fdr_available:
        return "proxy_plus_controls_plus_fdr"
    if proxy_supported and control_supported:
        return "proxy_plus_controls"
    if control_supported:
        return "controls_only"
    return "insufficient"


def _build_claims(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tiers = {row["dataset"]: row["evidence_tier"] for row in rows}
    all_controls = all(row["control_ranking_status"] == "supported" for row in rows)
    all_fdr = all(row["fdr_scope"] == "available_group_summary" for row in rows)
    lymph_curated = tiers.get("lymph_node") == "key_curated_validation"
    proxy_others = all(tiers.get(dataset) in {"proxy_plus_controls_plus_fdr", "proxy_plus_controls"} for dataset in ["tonsil", "melanoma"])
    return [
        {
            "claim_id": "P5_XT1_three_tissue_validation_scope",
            "status": "supported" if all_controls else "partial",
            "allowed_wording": "Lymph node, tonsil, and melanoma all have positive/negative control separation evidence.",
            "disallowed_wording": "All three tissues have native complete ground truth.",
            "evidence_value": "control_supported=" + str(all_controls),
            "limitation": "Control labels are injected or prior controls, not native complete CCC truth.",
        },
        {
            "claim_id": "P5_XT2_downstream_scope_guardrail",
            "status": "supported" if lymph_curated and proxy_others else "partial",
            "allowed_wording": "Lymph node has key curated downstream validation; tonsil and melanoma currently support receiver-response proxy or niche-control validation.",
            "disallowed_wording": "Current results prove global cross-tissue curated downstream validation.",
            "evidence_value": "tiers=" + "|".join(f"{dataset}:{tier}" for dataset, tier in sorted(tiers.items())),
            "limitation": "Cross-tissue downstream claim must remain guarded.",
        },
        {
            "claim_id": "P5_XT3_group_fdr_scope",
            "status": "supported" if all_fdr else "partial",
            "allowed_wording": "Group-level FDR outputs are available for the three real-background tissue panels, with the strongest 500-permutation key panel in lymph node.",
            "disallowed_wording": "All tissues have final 500-permutation edge-level FDR discoveries.",
            "evidence_value": "group_fdr_available=" + str(all_fdr),
            "limitation": "Only the designated lymph-node key panel reaches 500 permutations; other tissue panels are lower-depth/formal support.",
        },
    ]


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


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "1.0", "yes", "y", "supported"}


def _int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
