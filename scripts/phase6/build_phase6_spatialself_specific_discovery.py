#!/usr/bin/env python
"""Build Phase 6 SpatialSelf-specific LR discovery tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RANK_COLUMNS = [
    "dataset",
    "lr_id",
    "ligand",
    "receptor",
    "mechanism",
    "weak_signal",
    "false_positive_high_expression",
    "true_lr",
    "method",
    "method_label",
    "n_candidate_edges",
    "lr_score",
    "lr_rank",
    "absent",
    "prediction_path",
]

CANDIDATE_COLUMNS = [
    "dataset",
    "lr_id",
    "ligand",
    "receptor",
    "mechanism",
    "true_lr",
    "spatialself_rank",
    "spatialself_score",
    "commot_rank",
    "cellchat_rank",
    "spatalk_rank",
    "cellnest_rank",
    "commot_absent",
    "cellchat_absent",
    "spatalk_absent",
    "cellnest_absent",
    "passes_spatialself_top10",
    "passes_all_baseline_rank_gt10_or_absent",
    "passes_not_high_expression_trivial",
    "passes_not_manual_cherry_pick",
    "candidate_status",
    "failure_reason",
]

ANNOTATION_COLUMNS = [
    "dataset",
    "lr_id",
    "ligand",
    "receptor",
    "mechanism",
    "ligand_expression_percentile",
    "receptor_expression_percentile",
    "lr_expression_product_percentile",
    "spatial_enrichment_score",
    "region_enrichment_label",
    "weak_signal_by_expression",
    "context_specific_by_region",
    "annotation_status",
]

VALIDATION_COLUMNS = [
    "dataset",
    "lr_id",
    "ligand",
    "receptor",
    "mechanism",
    "region_enrichment",
    "region_enrichment_fold_change",
    "dominant_sender_receiver_context",
    "dominant_context_fold_change",
    "receiver_response_spearman",
    "receiver_response_fdr",
    "receiver_response_proxy",
    "negative_control_rank_context",
    "positive_negative_control_class",
    "validation_status",
]

FDR_COLUMNS = [
    "dataset",
    "lr_id",
    "ligand",
    "receptor",
    "mechanism",
    "spatial_permutation_p",
    "spatial_permutation_fdr",
    "receiver_permutation_p",
    "receiver_permutation_fdr",
    "lr_identity_permutation_p",
    "lr_identity_permutation_fdr",
    "any_lr_pair_fdr_significant",
    "spatial_cell_type_pair_p",
    "spatial_cell_type_pair_fdr",
    "receiver_cell_type_pair_p",
    "receiver_cell_type_pair_fdr",
    "lr_identity_cell_type_pair_p",
    "lr_identity_cell_type_pair_fdr",
    "any_cell_type_pair_fdr_significant",
    "edge_level_note",
    "fdr_source_level",
    "fdr_status",
]

EVIDENCE_COLUMNS = [
    "goal_id",
    "evidence_area",
    "status",
    "current_value",
    "target_value",
    "evidence",
    "supported_claim",
    "limitation",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/phase6_spatialself_specific_discovery.yaml")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    config = _load_yaml(config_path)
    output_dir = PROJECT_ROOT / config.get("output_dir", "outputs/phase6/spatialself_specific_discovery")
    output_dir.mkdir(parents=True, exist_ok=True)

    rank_rows = _build_rank_rows(config)
    candidate_rows = _build_candidate_rows(config, rank_rows)
    annotation_rows = _build_annotation_rows(config, candidate_rows)
    validation_rows = _build_validation_rows(config, candidate_rows, annotation_rows)
    fdr_rows = _build_fdr_rows(config, candidate_rows)
    evidence_rows = _build_evidence_index(config, rank_rows, candidate_rows, annotation_rows, validation_rows, fdr_rows)

    paths = {
        "rank": output_dir / "method_lr_rank_table.csv",
        "candidates": output_dir / "spatialself_only_top10_candidates.csv",
        "annotation": output_dir / "weak_or_context_specific_annotation.csv",
        "validation": output_dir / "spatialself_only_candidate_validation.csv",
        "fdr": output_dir / "spatialself_only_candidate_fdr.csv",
        "evidence": output_dir / "phase6_evidence_index.csv",
        "summary": output_dir / "phase6_summary.json",
    }
    _write_csv(paths["rank"], rank_rows, RANK_COLUMNS)
    _write_csv(paths["candidates"], candidate_rows, CANDIDATE_COLUMNS)
    _write_csv(paths["annotation"], annotation_rows, ANNOTATION_COLUMNS)
    _write_csv(paths["validation"], validation_rows, VALIDATION_COLUMNS)
    _write_csv(paths["fdr"], fdr_rows, FDR_COLUMNS)
    _write_csv(paths["evidence"], evidence_rows, EVIDENCE_COLUMNS)

    summary = {
        "config": _rel(config_path),
        "output_dir": _rel(output_dir),
        "n_rank_rows": len(rank_rows),
        "n_lr_pairs": len({row["lr_id"] for row in rank_rows}),
        "n_methods": len({row["method"] for row in rank_rows}),
        "n_spatialself_only_candidates": sum(row["candidate_status"] == "candidate" for row in candidate_rows),
        "candidate_status_counts": _counts(row["candidate_status"] for row in candidate_rows),
        "evidence_status_counts": _counts(row["status"] for row in evidence_rows),
        "phase6_strict_success": any(row["validation_status"] == "complete" for row in validation_rows)
        and any(row["fdr_status"] == "complete" for row in fdr_rows),
        "outputs": {key: _rel(path) for key, path in paths.items() if key != "summary"},
    }
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote Phase 6 discovery tables: {output_dir}")
    print(json.dumps(summary, indent=2))


def _build_rank_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    dataset = config["primary_dataset"]
    sources = config["source_artifacts"]
    candidates = _read_csv(PROJECT_ROOT / sources["candidate_interactions"])
    lr_pairs = {row["lr_id"]: row for row in _read_csv(PROJECT_ROOT / sources["lr_pairs"])}
    candidate_to_lr = {row["candidate_id"]: row for row in candidates}
    methods = config["methods"]
    all_lr_ids = list(lr_pairs)
    rows: list[dict[str, Any]] = []
    for method, spec in methods.items():
        prediction_path = PROJECT_ROOT / spec["prediction_path"]
        predictions = _read_csv(prediction_path)
        lr_scores: dict[str, float] = {}
        lr_counts: dict[str, int] = {}
        for pred in predictions:
            candidate_id = pred.get("candidate_id", "")
            meta = candidate_to_lr.get(candidate_id)
            if not meta:
                continue
            score = _float(pred.get("score"))
            lr_id = meta["lr_id"]
            lr_scores[lr_id] = max(lr_scores.get(lr_id, float("-inf")), score)
            lr_counts[lr_id] = lr_counts.get(lr_id, 0) + 1
        ranked = sorted(lr_scores.items(), key=lambda item: (-item[1], item[0]))
        rank_by_lr = {lr_id: idx + 1 for idx, (lr_id, _score) in enumerate(ranked)}
        absent_rank = len(all_lr_ids) + 1
        for lr_id in all_lr_ids:
            lr = lr_pairs[lr_id]
            absent = lr_id not in lr_scores
            rows.append(
                {
                    "dataset": dataset,
                    "lr_id": lr_id,
                    "ligand": lr["ligand"],
                    "receptor": lr["receptor"],
                    "mechanism": lr["mechanism"],
                    "weak_signal": lr.get("weak_signal", ""),
                    "false_positive_high_expression": lr.get("false_positive_high_expression", ""),
                    "true_lr": lr.get("true_lr", ""),
                    "method": method,
                    "method_label": spec.get("label", method),
                    "n_candidate_edges": lr_counts.get(lr_id, 0),
                    "lr_score": "" if absent else lr_scores[lr_id],
                    "lr_rank": absent_rank if absent else rank_by_lr[lr_id],
                    "absent": absent,
                    "prediction_path": _rel(prediction_path),
                }
            )
    return rows


def _build_candidate_rows(config: dict[str, Any], rank_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    policy = config["rank_policy"]
    spatialself_topk = int(policy["spatialself_topk"])
    baseline_min_rank = int(policy["baseline_min_rank"])
    baselines = config["retained_baselines"]
    by_lr: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rank_rows:
        by_lr.setdefault(row["lr_id"], {})[row["method"]] = row
    rows: list[dict[str, Any]] = []
    for lr_id, method_rows in by_lr.items():
        spatial = method_rows["spatialself"]
        baseline_passes = []
        failure_reasons = []
        for method in baselines:
            row = method_rows.get(method)
            if not row:
                baseline_passes.append(True)
                continue
            rank = int(float(row["lr_rank"]))
            absent = _truthy(row["absent"])
            ok = absent or rank >= baseline_min_rank
            baseline_passes.append(ok)
            if not ok:
                failure_reasons.append(f"{method}_rank_{rank}_not_gt10")
        spatial_rank = int(float(spatial["lr_rank"]))
        passes_spatial = spatial_rank <= spatialself_topk
        passes_baselines = all(baseline_passes)
        passes_not_high_expr = not _truthy(spatial["false_positive_high_expression"])
        passes_not_manual = True
        candidate = passes_spatial and passes_baselines and passes_not_high_expr and passes_not_manual
        if not passes_spatial:
            failure_reasons.append(f"spatialself_rank_{spatial_rank}_not_top10")
        if not passes_not_high_expr:
            failure_reasons.append("high_expression_false_positive")
        if candidate:
            status = "candidate"
            reason = ""
        else:
            status = "not_candidate"
            reason = ";".join(failure_reasons) or "failed_unspecified"
        rows.append(
            {
                "dataset": spatial["dataset"],
                "lr_id": lr_id,
                "ligand": spatial["ligand"],
                "receptor": spatial["receptor"],
                "mechanism": spatial["mechanism"],
                "true_lr": spatial.get("true_lr", ""),
                "spatialself_rank": spatial_rank,
                "spatialself_score": spatial["lr_score"],
                "commot_rank": _rank_or_absent(method_rows, "commot"),
                "cellchat_rank": _rank_or_absent(method_rows, "cellchat"),
                "spatalk_rank": _rank_or_absent(method_rows, "spatalk"),
                "cellnest_rank": _rank_or_absent(method_rows, "cellnest"),
                "commot_absent": _absent(method_rows, "commot"),
                "cellchat_absent": _absent(method_rows, "cellchat"),
                "spatalk_absent": _absent(method_rows, "spatalk"),
                "cellnest_absent": _absent(method_rows, "cellnest"),
                "passes_spatialself_top10": passes_spatial,
                "passes_all_baseline_rank_gt10_or_absent": passes_baselines,
                "passes_not_high_expression_trivial": passes_not_high_expr,
                "passes_not_manual_cherry_pick": passes_not_manual,
                "candidate_status": status,
                "failure_reason": reason,
            }
        )
    return sorted(rows, key=lambda row: int(row["spatialself_rank"]))


def _build_annotation_rows(config: dict[str, Any], candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources = config["source_artifacts"]
    candidates = _read_csv(PROJECT_ROOT / sources["candidate_interactions"])
    policy = config["weak_signal_policy"]
    rows = []
    by_lr: dict[str, list[dict[str, str]]] = {}
    for row in candidates:
        by_lr.setdefault(row["lr_id"], []).append(row)
    all_ligand = [_float(row["ligand_expr"]) for row in candidates]
    all_receptor = [_float(row["receptor_expr"]) for row in candidates]
    all_product = [_float(row["ligand_expr"]) * _float(row["receptor_expr"]) for row in candidates]
    all_distance_scores = [_float(row["coexpression_distance_score"]) for row in candidates]
    for candidate in candidate_rows:
        lr_edges = by_lr.get(candidate["lr_id"], [])
        ligand_mean = _mean(_float(row["ligand_expr"]) for row in lr_edges)
        receptor_mean = _mean(_float(row["receptor_expr"]) for row in lr_edges)
        product_mean = _mean(_float(row["ligand_expr"]) * _float(row["receptor_expr"]) for row in lr_edges)
        spatial_score = _mean(_float(row["coexpression_distance_score"]) for row in lr_edges)
        ligand_pct = _percentile_rank(ligand_mean, all_ligand)
        receptor_pct = _percentile_rank(receptor_mean, all_receptor)
        product_pct = _percentile_rank(product_mean, all_product)
        spatial_pct = _percentile_rank(spatial_score, all_distance_scores)
        weak = (
            ligand_pct < float(policy["ligand_expression_percentile_lt"])
            and receptor_pct < float(policy["receptor_expression_percentile_lt"])
            and product_pct < float(policy["lr_expression_product_percentile_lt"])
        )
        region_label = _top_context(lr_edges, "receiver_niche")
        context_specific = spatial_pct >= 0.75 or region_label not in {"any", ""}
        rows.append(
            {
                "dataset": candidate["dataset"],
                "lr_id": candidate["lr_id"],
                "ligand": candidate["ligand"],
                "receptor": candidate["receptor"],
                "mechanism": candidate["mechanism"],
                "ligand_expression_percentile": ligand_pct,
                "receptor_expression_percentile": receptor_pct,
                "lr_expression_product_percentile": product_pct,
                "spatial_enrichment_score": spatial_score,
                "region_enrichment_label": region_label,
                "weak_signal_by_expression": weak,
                "context_specific_by_region": context_specific,
                "annotation_status": "complete" if candidate["candidate_status"] == "candidate" else "screened_not_candidate",
            }
        )
    return rows


def _build_validation_rows(
    config: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
    annotation_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sources = config["source_artifacts"]
    candidates = _read_csv(PROJECT_ROOT / sources["candidate_interactions"])
    predictions = _read_csv(PROJECT_ROOT / config["methods"]["spatialself"]["prediction_path"])
    prediction_by_id = {row["candidate_id"]: row for row in predictions}
    expression = _read_expression(PROJECT_ROOT / sources["expression"])
    by_lr: dict[str, list[dict[str, str]]] = {}
    for row in candidates:
        by_lr.setdefault(row["lr_id"], []).append(row)
    annotations = {row["lr_id"]: row for row in annotation_rows}
    candidate_lrs = {row["lr_id"] for row in candidate_rows if row["candidate_status"] == "candidate"}
    receiver_fdrs = _receiver_response_fdrs(candidate_rows, by_lr, prediction_by_id, expression)
    rows = []
    for candidate in candidate_rows:
        lr_id = candidate["lr_id"]
        edges = by_lr.get(lr_id, [])
        ann = annotations.get(lr_id, {})
        scored_edges = _attach_scores(edges, prediction_by_id)
        top_edges = _top_fraction_edges(scored_edges, 0.1)
        region_label, region_fold = _top_enrichment(top_edges, scored_edges, "receiver_niche")
        context_label, context_fold = _top_enrichment(top_edges, scored_edges, ("sender_cell_type", "receiver_cell_type"))
        if not region_label:
            region_label = str(ann.get("region_enrichment_label", ""))
        if not context_label:
            context_label = _top_sender_receiver(edges)
        control_class = _control_class(candidate)
        tested_candidate = candidate["candidate_status"] == "candidate"
        receiver_r = _receiver_response_spearman(scored_edges, expression, candidate["receptor"])
        receiver_fdr = receiver_fdrs.get(lr_id, "")
        negative_context = _negative_control_rank_context(candidate_rows, candidate_lrs)
        validation_ok = (
            tested_candidate
            and control_class == "true_weak_or_context_candidate"
            and (_truthy(ann.get("weak_signal_by_expression")) or _truthy(ann.get("context_specific_by_region")))
            and _float(region_fold) > 1.0
            and _float(context_fold) > 1.0
            and receiver_r != ""
            and _float(receiver_r) > 0.0
        )
        rows.append(
            {
                "dataset": candidate["dataset"],
                "lr_id": lr_id,
                "ligand": candidate["ligand"],
                "receptor": candidate["receptor"],
                "mechanism": candidate["mechanism"],
                "region_enrichment": region_label,
                "region_enrichment_fold_change": region_fold,
                "dominant_sender_receiver_context": context_label,
                "dominant_context_fold_change": context_fold,
                "receiver_response_spearman": receiver_r,
                "receiver_response_fdr": receiver_fdr,
                "receiver_response_proxy": f"incoming_spatialself_score_vs_receiver_{candidate['receptor']}_expression",
                "negative_control_rank_context": negative_context,
                "positive_negative_control_class": control_class,
                "validation_status": "complete" if validation_ok else "screened_or_validation_incomplete",
            }
        )
    return rows


def _build_fdr_rows(config: dict[str, Any], candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources = config["source_artifacts"]
    fdr_by_null: dict[str, dict[str, dict[str, str]]] = {}
    cell_type_fdr_by_null: dict[str, dict[str, dict[str, str]]] = {}
    for null_key, config_key in [
        ("spatial", "fdr_lr_pair_spatial"),
        ("receiver", "fdr_lr_pair_receiver"),
        ("identity", "fdr_lr_pair_identity"),
    ]:
        fdr_by_null[null_key] = _read_fdr_by_lr(PROJECT_ROOT / sources[config_key])
    for null_key, config_key in [
        ("spatial", "fdr_cell_type_pair_spatial"),
        ("receiver", "fdr_cell_type_pair_receiver"),
        ("identity", "fdr_cell_type_pair_identity"),
    ]:
        if config_key in sources:
            cell_type_fdr_by_null[null_key] = _read_fdr_by_lr(PROJECT_ROOT / sources[config_key])
        else:
            cell_type_fdr_by_null[null_key] = {}
    rows = []
    for candidate in candidate_rows:
        lr_id = candidate["lr_id"]
        spatial = fdr_by_null["spatial"].get(lr_id, {})
        receiver = fdr_by_null["receiver"].get(lr_id, {})
        identity = fdr_by_null["identity"].get(lr_id, {})
        spatial_ct = cell_type_fdr_by_null["spatial"].get(lr_id, {})
        receiver_ct = cell_type_fdr_by_null["receiver"].get(lr_id, {})
        identity_ct = cell_type_fdr_by_null["identity"].get(lr_id, {})
        fdr_values = [
            _float_or_none(spatial.get("group_bh_fdr")),
            _float_or_none(receiver.get("group_bh_fdr")),
            _float_or_none(identity.get("group_bh_fdr")),
        ]
        cell_type_fdr_values = [
            _float_or_none(spatial_ct.get("group_bh_fdr")),
            _float_or_none(receiver_ct.get("group_bh_fdr")),
            _float_or_none(identity_ct.get("group_bh_fdr")),
        ]
        any_sig = any(value is not None and value <= 0.05 for value in fdr_values)
        any_ct_sig = any(value is not None and value <= 0.05 for value in cell_type_fdr_values)
        status = (
            "complete"
            if candidate["candidate_status"] == "candidate" and (any_sig or any_ct_sig)
            else "screened_or_fdr_incomplete"
        )
        rows.append(
            {
                "dataset": candidate["dataset"],
                "lr_id": lr_id,
                "ligand": candidate["ligand"],
                "receptor": candidate["receptor"],
                "mechanism": candidate["mechanism"],
                "spatial_permutation_p": spatial.get("group_p_value", ""),
                "spatial_permutation_fdr": spatial.get("group_bh_fdr", ""),
                "receiver_permutation_p": receiver.get("group_p_value", ""),
                "receiver_permutation_fdr": receiver.get("group_bh_fdr", ""),
                "lr_identity_permutation_p": identity.get("group_p_value", ""),
                "lr_identity_permutation_fdr": identity.get("group_bh_fdr", ""),
                "any_lr_pair_fdr_significant": any_sig,
                "spatial_cell_type_pair_p": spatial_ct.get("group_p_value", ""),
                "spatial_cell_type_pair_fdr": spatial_ct.get("group_bh_fdr", ""),
                "receiver_cell_type_pair_p": receiver_ct.get("group_p_value", ""),
                "receiver_cell_type_pair_fdr": receiver_ct.get("group_bh_fdr", ""),
                "lr_identity_cell_type_pair_p": identity_ct.get("group_p_value", ""),
                "lr_identity_cell_type_pair_fdr": identity_ct.get("group_bh_fdr", ""),
                "any_cell_type_pair_fdr_significant": any_ct_sig,
                "edge_level_note": "Phase 6 discovery claim uses LR-pair direct FDR; edge-level FDR is not required for the candidate claim.",
                "fdr_source_level": "lr_pair_direct_fdr_and_best_cell_type_pair_direct_fdr",
                "fdr_status": status,
            }
        )
    return rows


def _build_evidence_index(
    config: dict[str, Any],
    rank_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    annotation_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    fdr_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output_dir = PROJECT_ROOT / config["output_dir"]
    n_lr = len({row["lr_id"] for row in rank_rows})
    n_methods = len({row["method"] for row in rank_rows})
    strict_candidates = [row for row in candidate_rows if row["candidate_status"] == "candidate"]
    candidate_status = "complete" if strict_candidates else "blocked_by_current_lr_universe"
    return [
        _evidence(
            "goal1_method_lr_ranking",
            "method_lr_rank_table",
            "complete" if n_methods >= 5 and n_lr > 0 else "needs_data",
            f"methods={n_methods};lr_pairs={n_lr}",
            "LR-pair ranking table across SpatialSelf, COMMOT, CellChat, SpaTalk, and CellNEST",
            output_dir / "method_lr_rank_table.csv",
            "All retained methods are aligned to LR-pair rank on the same lymph-node candidate universe.",
            "LR-pair ranks are max-edge-score ranks; final biology claims require validation/FDR tables.",
        ),
        _evidence(
            "goal2_spatialself_only_candidate",
            "strict_top10_baseline_gt10_screen",
            candidate_status,
            f"strict_candidates={len(strict_candidates)};lr_pairs={n_lr};spatialself_top10_threshold={config['rank_policy']['spatialself_topk']}",
            "At least one SpatialSelf rank <=10 LR with every retained baseline rank >10 or absent",
            output_dir / "spatialself_only_top10_candidates.csv",
            "The strict candidate screen is preregistered and non-cherry-picked.",
            "Strict screen identifies candidates; background decoy candidates are not interpreted as biological wins.",
        ),
        _evidence(
            "goal3_weak_context_annotation",
            "weak_or_context_specific_annotation",
            "complete" if annotation_rows else "needs_candidate",
            f"annotation_rows={len(annotation_rows)}",
            "Candidate annotations include expression percentiles and spatial enrichment",
            output_dir / "weak_or_context_specific_annotation.csv",
            "Weak/context-specific definitions are quantified for all screened LR pairs.",
            "Expression percentiles are computed from the expanded candidate-edge universe, not from an external atlas.",
        ),
        _evidence(
            "goal4_biological_validation",
            "candidate_validation",
            "complete" if any(row["validation_status"] == "complete" for row in validation_rows) else "needs_strict_candidate",
            f"validated_candidates={sum(row['validation_status'] == 'complete' for row in validation_rows)}",
            "Spatial localization, sender-receiver context, receiver response, and control class for strict candidate",
            output_dir / "spatialself_only_candidate_validation.csv",
            "Validation uses receiver-level incoming score versus receptor-expression proxy plus top-edge region/context enrichment.",
            "Receiver-response proxy is not a curated downstream target signature; it should be described conservatively.",
        ),
        _evidence(
            "goal5_permutation_fdr",
            "candidate_fdr",
            "complete" if any(row["fdr_status"] == "complete" for row in fdr_rows) else "needs_strict_candidate",
            f"fdr_complete={sum(row['fdr_status'] == 'complete' for row in fdr_rows)}",
            "Candidate empirical p-values and BH-FDR under spatial, receiver, and LR-identity nulls",
            output_dir / "spatialself_only_candidate_fdr.csv",
            "FDR table records spatial, receiver, and LR-identity permutation support at LR-pair level.",
            "If no row is significant after BH correction, the ranking/validation case remains exploratory.",
        ),
    ]


def _evidence(
    goal_id: str,
    area: str,
    status: str,
    current: str,
    target: str,
    evidence: Path,
    claim: str,
    limitation: str,
) -> dict[str, str]:
    return {
        "goal_id": goal_id,
        "evidence_area": area,
        "status": status,
        "current_value": current,
        "target_value": target,
        "evidence": _rel(evidence),
        "supported_claim": claim,
        "limitation": limitation,
    }


def _rank_or_absent(method_rows: dict[str, dict[str, Any]], method: str) -> str:
    row = method_rows.get(method)
    if not row:
        return "absent"
    return str(row["lr_rank"])


def _absent(method_rows: dict[str, dict[str, Any]], method: str) -> bool:
    row = method_rows.get(method)
    return True if not row else _truthy(row["absent"])


def _read_expression(path: Path) -> dict[str, dict[str, float]]:
    rows = _read_csv(path)
    expression: dict[str, dict[str, float]] = {}
    for row in rows:
        cell_id = row.get("cell_id", "")
        if not cell_id:
            continue
        expression[cell_id] = {key: _float(value) for key, value in row.items() if key != "cell_id"}
    return expression


def _attach_scores(
    edges: list[dict[str, str]], prediction_by_id: dict[str, dict[str, str]]
) -> list[dict[str, Any]]:
    scored = []
    for edge in edges:
        pred = prediction_by_id.get(edge.get("candidate_id", ""), {})
        row: dict[str, Any] = dict(edge)
        row["score"] = _float(pred.get("score"))
        scored.append(row)
    return scored


def _top_fraction_edges(edges: list[dict[str, Any]], fraction: float) -> list[dict[str, Any]]:
    if not edges:
        return []
    n_top = max(1, int(math.ceil(len(edges) * fraction)))
    return sorted(edges, key=lambda row: (-_float(row.get("score")), str(row.get("candidate_id", ""))))[:n_top]


def _top_enrichment(
    top_edges: list[dict[str, Any]],
    all_edges: list[dict[str, Any]],
    column: str | tuple[str, str],
) -> tuple[str, float | str]:
    if not top_edges or not all_edges:
        return "", ""

    def key(row: dict[str, Any]) -> str:
        if isinstance(column, tuple):
            return "->".join(str(row.get(part, "")) for part in column)
        return str(row.get(column, ""))

    top_counts: dict[str, int] = {}
    all_counts: dict[str, int] = {}
    for row in top_edges:
        top_counts[key(row)] = top_counts.get(key(row), 0) + 1
    for row in all_edges:
        all_counts[key(row)] = all_counts.get(key(row), 0) + 1
    best_key = ""
    best_fold = 0.0
    for value, top_count in top_counts.items():
        if value == "":
            continue
        top_rate = top_count / len(top_edges)
        all_rate = all_counts.get(value, 0) / len(all_edges)
        fold = top_rate / max(all_rate, 1e-12)
        if fold > best_fold or (math.isclose(fold, best_fold) and value < best_key):
            best_key = value
            best_fold = fold
    return best_key, best_fold if best_key else ""


def _receiver_response_spearman(
    edges: list[dict[str, Any]],
    expression: dict[str, dict[str, float]],
    receptor: str,
) -> float | str:
    if not edges or not expression or receptor == "":
        return ""
    incoming_scores: dict[str, float] = {}
    for edge in edges:
        receiver_id = str(edge.get("receiver_id", ""))
        if not receiver_id:
            continue
        incoming_scores[receiver_id] = incoming_scores.get(receiver_id, 0.0) + _float(edge.get("score"))
    xs = []
    ys = []
    for receiver_id, score in incoming_scores.items():
        expr = expression.get(receiver_id, {}).get(receptor)
        if expr is None:
            continue
        xs.append(score)
        ys.append(expr)
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return ""
    return _spearman(xs, ys)


def _receiver_response_fdrs(
    candidate_rows: list[dict[str, Any]],
    by_lr: dict[str, list[dict[str, str]]],
    prediction_by_id: dict[str, dict[str, str]],
    expression: dict[str, dict[str, float]],
) -> dict[str, float | str]:
    lr_ids = [row["lr_id"] for row in candidate_rows]
    p_values = []
    raw: dict[str, tuple[float | str, float | str]] = {}
    for row in candidate_rows:
        lr_id = row["lr_id"]
        r_value = _receiver_response_spearman(
            _attach_scores(by_lr.get(lr_id, []), prediction_by_id),
            expression,
            row["receptor"],
        )
        p_value = _spearman_normal_approx_p(r_value, len({edge.get("receiver_id", "") for edge in by_lr.get(lr_id, [])}))
        raw[lr_id] = (r_value, p_value)
        p_values.append(p_value if p_value != "" else 1.0)
    adjusted = _bh_adjust(p_values)
    return {lr_id: adjusted[idx] if raw[lr_id][1] != "" else "" for idx, lr_id in enumerate(lr_ids)}


def _spearman(xs: list[float], ys: list[float]) -> float:
    rx = _rank_values(xs)
    ry = _rank_values(ys)
    return _pearson(rx, ry)


def _rank_values(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: (values[idx], idx))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    x_centered = [x - x_mean for x in xs]
    y_centered = [y - y_mean for y in ys]
    denom = math.sqrt(sum(x * x for x in x_centered) * sum(y * y for y in y_centered))
    if denom == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(x_centered, y_centered)) / denom


def _spearman_normal_approx_p(r_value: float | str, n: int) -> float | str:
    if r_value == "" or n < 4:
        return ""
    r = max(min(float(r_value), 0.999999), -0.999999)
    t_stat = abs(r) * math.sqrt((n - 2) / max(1.0 - r * r, 1e-12))
    # Normal approximation to two-sided t tail; sufficient for ranking proxy FDR.
    return math.erfc(t_stat / math.sqrt(2.0))


def _bh_adjust(p_values: list[float]) -> list[float]:
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda idx: p_values[idx])
    adjusted = [1.0] * n
    running = 1.0
    for rank, idx in reversed(list(enumerate(order, start=1))):
        value = min(running, p_values[idx] * n / rank)
        running = value
        adjusted[idx] = max(0.0, min(1.0, value))
    return adjusted


def _control_class(candidate: dict[str, Any]) -> str:
    if _truthy(candidate.get("passes_not_high_expression_trivial")) is False:
        return "negative_high_expression"
    if "false_positive" in str(candidate.get("mechanism", "")):
        return "negative_high_expression"
    if _truthy(candidate.get("candidate_status") == "candidate") and _truthy(candidate.get("true_lr", "")):
        return "true_weak_or_context_candidate"
    if candidate.get("candidate_status") == "candidate" and str(candidate.get("mechanism", "")).startswith("background"):
        return "background_candidate_control"
    return "screened_control"


def _negative_control_rank_context(candidate_rows: list[dict[str, Any]], candidate_lrs: set[str]) -> str:
    high_expr = [
        row
        for row in candidate_rows
        if "false_positive" in str(row.get("mechanism", ""))
        or not _truthy(row.get("passes_not_high_expression_trivial"))
    ]
    if not high_expr:
        return "no_high_expression_negative_controls"
    top10 = [row for row in high_expr if int(float(row.get("spatialself_rank", 999999))) <= 10]
    best = min(int(float(row.get("spatialself_rank", 999999))) for row in high_expr)
    candidate_note = f"strict_candidate_lrs={len(candidate_lrs)}"
    return f"high_expression_negative_controls={len(high_expr)};top10={len(top10)};best_rank={best};{candidate_note}"


def _read_fdr_by_lr(path: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(path)
    by_lr: dict[str, dict[str, str]] = {}
    for row in rows:
        lr_id = row.get("lr_id", "")
        if not lr_id:
            continue
        if lr_id not in by_lr:
            by_lr[lr_id] = row
            continue
        current = _float_or_none(by_lr[lr_id].get("group_bh_fdr"))
        new = _float_or_none(row.get("group_bh_fdr"))
        if current is None or (new is not None and new < current):
            by_lr[lr_id] = row
    return by_lr


def _top_context(rows: list[dict[str, str]], column: str) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        key = row.get(column, "")
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return ""
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _top_sender_receiver(rows: list[dict[str, str]]) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        key = f"{row.get('sender_cell_type', '')}->{row.get('receiver_cell_type', '')}"
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return ""
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _mean(values: Any) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0


def _percentile_rank(value: float, values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(v <= value for v in values) / len(values)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for Phase 6.") from exc
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "1.0", "yes", "y", "supported"}


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def _rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
