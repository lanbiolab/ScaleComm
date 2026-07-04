#!/usr/bin/env python
"""Audit Phase 7 Goal 4 cross-background SpatialSelf-specific discovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASETS = {
    "tonsil": {
        "dataset_id": "human_tonsil_slide_tags_replicate_00",
        "dataset_dir": "outputs/phase2_revised/goal4_real_background_v2/formal_datasets/human_tonsil_slide_tags_replicate_00",
        "fdr_dir": "outputs/phase2_revised/goal5_real_background_fdr_panel/real_background_v2_formal_tonsil",
        "downstream_dir": "outputs/phase2_revised/goal1_curated_downstream_panel/real_background_v2_formal_tonsil",
    },
    "melanoma": {
        "dataset_id": "human_melanoma_multiome_replicate_00",
        "dataset_dir": "outputs/phase2_revised/goal4_real_background_v2/formal_datasets/human_melanoma_multiome_replicate_00",
        "fdr_dir": "outputs/phase2_revised/goal5_real_background_fdr_panel/real_background_v2_formal_melanoma",
        "downstream_dir": "outputs/phase2_revised/goal1_curated_downstream_panel/real_background_v2_formal_melanoma",
    },
}

METHOD_SPECS = {
    "spatialself": {
        "label": "SpatialSelf",
        "path_template": "outputs/phase2_revised/goal2_formal_baseline_panel/real_background_v2_formal/spatialself_wavelet_edge_prior/{dataset_id}__spatialself_wavelet_predictions.csv",
    },
    "commot": {
        "label": "COMMOT",
        "path_template": "outputs/phase2_revised/goal2_formal_baseline_panel/real_background_v2_formal/commot/{dataset_id}__commot_predictions.csv",
    },
    "cellchat": {
        "label": "CellChat",
        "path_template": "outputs/phase2_revised/goal2_formal_baseline_panel/real_background_v2_formal/cellchat/{dataset_id}__cellchat_predictions.csv",
    },
    "spatalk": {
        "label": "SpaTalk",
        "path_template": "outputs/phase2_revised/goal2_formal_baseline_panel/real_background_v2_formal/spatalk/{dataset_id}__spatalk_predictions.csv",
    },
    "cellnest": {
        "label": "CellNEST",
        "path_template": "outputs/phase2_revised/goal2_formal_baseline_panel_extended/real_background_v2_formal/cellnest/{dataset_id}__cellnest_predictions.csv",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/phase7_functional_cross_platform_validation.yaml",
    )
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    goal_cfg = config["goals"]["goal4_cross_background_spatialself_specific_discovery"]
    goal_dir = PROJECT_ROOT / goal_cfg["output_dir"]
    goal_dir.mkdir(parents=True, exist_ok=True)

    rank_rows = []
    candidate_rows = []
    validation_rows = []
    fdr_rows = []
    audit_rows = []

    for dataset_key in goal_cfg.get("target_backgrounds", ["tonsil", "melanoma"]):
        dataset = DATASETS[dataset_key]
        dataset_dir = PROJECT_ROOT / dataset["dataset_dir"]
        candidates = pd.read_csv(dataset_dir / "candidate_interactions.csv")
        lr_pairs = pd.read_csv(dataset_dir / "lr_pairs.csv")
        cells = pd.read_csv(dataset_dir / "cells.csv")
        expression = pd.read_csv(dataset_dir / "expression.csv")

        dataset_rank = _rank_methods(dataset_key, dataset, candidates, lr_pairs)
        rank_rows.append(dataset_rank)
        dataset_candidates = _screen_candidates(dataset_key, dataset_rank, lr_pairs, goal_cfg)
        candidate_rows.append(dataset_candidates)
        validation_rows.append(
            _validate_candidates(dataset_key, dataset, dataset_candidates, candidates, cells, expression)
        )
        fdr_rows.append(_collect_fdr(dataset_key, dataset, dataset_candidates))
        audit_rows.append(_audit_dataset(dataset_key, dataset_candidates, lr_pairs, goal_cfg))

    rank_table = pd.concat(rank_rows, ignore_index=True)
    candidates_table = pd.concat(candidate_rows, ignore_index=True)
    validation_table = pd.concat(validation_rows, ignore_index=True)
    fdr_table = pd.concat(fdr_rows, ignore_index=True)
    audit_table = pd.DataFrame(audit_rows)

    rank_table.to_csv(goal_dir / "phase7_cross_background_method_lr_rank_table.csv", index=False)
    candidates_table.to_csv(goal_dir / "phase7_cross_background_spatialself_only_candidates.csv", index=False)
    validation_table.to_csv(goal_dir / "phase7_cross_background_candidate_validation.csv", index=False)
    fdr_table.to_csv(goal_dir / "phase7_cross_background_candidate_fdr.csv", index=False)
    audit_table.to_csv(goal_dir / "phase7_cross_background_strict_screen_audit.csv", index=False)

    manifest = {
        "config": _rel(config_path),
        "goal_dir": _rel(goal_dir),
        "strict_success": bool((candidates_table["candidate_status"] == "strict_candidate").any()),
        "negative_evidence_recorded": bool((audit_table["audit_status"] == "blocked_by_lr_universe_size").any()),
        "interpretation": (
            "The current tonsil/melanoma formal panels contain only five LR pairs each, so the preregistered "
            "baseline-rank >10 criterion cannot be satisfied. This is recorded as negative evidence and motivates "
            "an expanded tonsil/melanoma LR universe before making a second SpatialSelf-specific discovery claim."
        ),
        "outputs": {
            "rank_table": _rel(goal_dir / "phase7_cross_background_method_lr_rank_table.csv"),
            "candidates": _rel(goal_dir / "phase7_cross_background_spatialself_only_candidates.csv"),
            "validation": _rel(goal_dir / "phase7_cross_background_candidate_validation.csv"),
            "fdr": _rel(goal_dir / "phase7_cross_background_candidate_fdr.csv"),
            "audit": _rel(goal_dir / "phase7_cross_background_strict_screen_audit.csv"),
        },
    }
    (goal_dir / "phase7_goal4_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(audit_table.to_string(index=False))


def _rank_methods(
    dataset_key: str,
    dataset: dict[str, str],
    candidates: pd.DataFrame,
    lr_pairs: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    lr_meta = lr_pairs.set_index("lr_id").to_dict("index")
    candidate_meta = candidates[["candidate_id", "lr_id"]].copy()
    for method, spec in METHOD_SPECS.items():
        pred_path = PROJECT_ROOT / spec["path_template"].format(dataset_id=dataset["dataset_id"])
        if pred_path.exists():
            pred = pd.read_csv(pred_path)
            merged = candidate_meta.merge(pred[["candidate_id", "score"]], on="candidate_id", how="left")
            lr_score = merged.groupby("lr_id", sort=False)["score"].max().reset_index()
            lr_score["absent"] = lr_score["score"].isna()
        else:
            lr_score = pd.DataFrame({"lr_id": lr_pairs["lr_id"], "score": np.nan, "absent": True})
        lr_score = lr_pairs[["lr_id"]].merge(lr_score, on="lr_id", how="left")
        lr_score["absent"] = lr_score["absent"].fillna(True)
        lr_score["score"] = pd.to_numeric(lr_score["score"], errors="coerce")
        scored = lr_score[~lr_score["absent"].astype(bool)].sort_values(["score", "lr_id"], ascending=[False, True])
        ranks = {lr_id: rank for rank, lr_id in enumerate(scored["lr_id"], start=1)}
        absent_rank = len(lr_pairs) + 1
        for row in lr_score.itertuples(index=False):
            meta = lr_meta[row.lr_id]
            rows.append(
                {
                    "dataset": dataset_key,
                    "lr_id": row.lr_id,
                    "ligand": meta["ligand"],
                    "receptor": meta["receptor"],
                    "mechanism": meta["mechanism"],
                    "weak_signal": bool(meta["weak_signal"]),
                    "false_positive_high_expression": bool(meta["false_positive_high_expression"]),
                    "true_lr": bool(meta["true_lr"]),
                    "method": method,
                    "method_label": spec["label"],
                    "lr_score": row.score,
                    "lr_rank": ranks.get(row.lr_id, absent_rank),
                    "absent": bool(row.absent),
                    "prediction_path": _rel(pred_path) if pred_path.exists() else "",
                }
            )
    return pd.DataFrame(rows)


def _screen_candidates(
    dataset_key: str,
    rank_table: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    goal_cfg: dict[str, Any],
) -> pd.DataFrame:
    strict = goal_cfg["strict_screen"]
    n_lr = len(lr_pairs)
    rows = []
    for lr_id, group in rank_table.groupby("lr_id", sort=False):
        spatial = group[group["method"] == "spatialself"].iloc[0]
        baseline_rows = group[group["method"].isin(goal_cfg["retained_baselines"])]
        baseline_pass = bool(((baseline_rows["lr_rank"] > strict["baseline_rank_gt_or_absent"]) | baseline_rows["absent"]).all())
        enough_rank_depth = n_lr > strict["baseline_rank_gt_or_absent"]
        not_high_expr = not bool(spatial["false_positive_high_expression"])
        weak_or_context = bool(spatial["weak_signal"]) or "niche" in str(spatial["mechanism"]) or "contact" in str(spatial["mechanism"])
        pass_spatial = int(spatial["lr_rank"]) <= strict["spatialself_rank_lte"]
        strict_candidate = pass_spatial and baseline_pass and enough_rank_depth and not_high_expr and weak_or_context
        reasons = []
        if not enough_rank_depth:
            reasons.append(f"lr_universe_size_{n_lr}_cannot_test_baseline_rank_gt_{strict['baseline_rank_gt_or_absent']}")
        if not baseline_pass:
            reasons.append("at_least_one_baseline_rank_not_gt10_or_absent")
        if not not_high_expr:
            reasons.append("high_expression_negative_control")
        if not weak_or_context:
            reasons.append("not_weak_or_context_specific")
        rows.append(
            {
                "dataset": dataset_key,
                "lr_id": lr_id,
                "ligand": spatial["ligand"],
                "receptor": spatial["receptor"],
                "mechanism": spatial["mechanism"],
                "weak_signal": bool(spatial["weak_signal"]),
                "true_lr": bool(spatial["true_lr"]),
                "spatialself_rank": int(spatial["lr_rank"]),
                "commot_rank": _method_rank(group, "commot"),
                "cellchat_rank": _method_rank(group, "cellchat"),
                "spatalk_rank": _method_rank(group, "spatalk"),
                "cellnest_rank": _method_rank(group, "cellnest"),
                "passes_spatialself_rank_lte10": pass_spatial,
                "passes_all_baseline_rank_gt10_or_absent": baseline_pass,
                "passes_lr_universe_rank_depth": enough_rank_depth,
                "passes_not_high_expression_trivial": not_high_expr,
                "passes_weak_or_context_specific": weak_or_context,
                "candidate_status": "strict_candidate" if strict_candidate else "not_strict_candidate",
                "failure_reason": ";".join(reasons),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "spatialself_rank", "lr_id"])


def _validate_candidates(
    dataset_key: str,
    dataset: dict[str, str],
    candidate_table: pd.DataFrame,
    candidates: pd.DataFrame,
    cells: pd.DataFrame,
    expression: pd.DataFrame,
) -> pd.DataFrame:
    pred_path = PROJECT_ROOT / METHOD_SPECS["spatialself"]["path_template"].format(dataset_id=dataset["dataset_id"])
    predictions = pd.read_csv(pred_path)
    scored = candidates.merge(predictions[["candidate_id", "score"]], on="candidate_id", how="left")
    downstream = _read_downstream(dataset, method="spatialself_wavelet_edge_prior")
    rows = []
    for row in candidate_table.itertuples(index=False):
        lr_edges = scored[scored["lr_id"] == row.lr_id].copy()
        top = lr_edges.sort_values("score", ascending=False).head(max(1, int(np.ceil(len(lr_edges) * 0.1))))
        region, region_fc = _top_enrichment(top, lr_edges, "receiver_niche")
        context, context_fc = _top_enrichment(top, lr_edges, ["sender_cell_type", "receiver_cell_type"])
        receptor_proxy_r = _receiver_proxy_spearman(lr_edges, expression, row.receptor)
        ds = downstream.get(row.lr_id, {})
        downstream_status = ds.get("status", "missing")
        downstream_r = ds.get("spearman_r", "")
        downstream_fdr = ds.get("bh_fdr", "")
        validation_status = (
            "strict_candidate_validated"
            if row.candidate_status == "strict_candidate"
            and float(region_fc or 0) > 1.0
            and float(context_fc or 0) > 1.0
            and (downstream_status == "tested" or receptor_proxy_r != "")
            else "screened_or_not_strict"
        )
        rows.append(
            {
                "dataset": dataset_key,
                "lr_id": row.lr_id,
                "ligand": row.ligand,
                "receptor": row.receptor,
                "mechanism": row.mechanism,
                "candidate_status": row.candidate_status,
                "region_enrichment": region,
                "region_enrichment_fold_change": region_fc,
                "dominant_sender_receiver_context": context,
                "dominant_context_fold_change": context_fc,
                "receiver_receptor_proxy_spearman": receptor_proxy_r,
                "downstream_target_status": downstream_status,
                "downstream_target_spearman": downstream_r,
                "downstream_target_fdr": downstream_fdr,
                "validation_status": validation_status,
            }
        )
    return pd.DataFrame(rows)


def _collect_fdr(dataset_key: str, dataset: dict[str, str], candidate_table: pd.DataFrame) -> pd.DataFrame:
    fdr_dir = PROJECT_ROOT / dataset["fdr_dir"]
    fdr_maps = {}
    for null in ["spatial_permutation", "receiver_permutation", "lr_identity_permutation"]:
        path = fdr_dir / f"{dataset['dataset_id']}__spatialself_wavelet_predictions__{null}__lr_pair_direct_fdr.csv"
        fdr_maps[null] = pd.read_csv(path).set_index("lr_id").to_dict("index") if path.exists() else {}
    rows = []
    for row in candidate_table.itertuples(index=False):
        out = {
            "dataset": dataset_key,
            "lr_id": row.lr_id,
            "ligand": row.ligand,
            "receptor": row.receptor,
            "mechanism": row.mechanism,
            "candidate_status": row.candidate_status,
        }
        sig = False
        for null, fmap in fdr_maps.items():
            item = fmap.get(row.lr_id, {})
            out[f"{null}_p_value"] = item.get("group_p_value", "")
            out[f"{null}_bh_fdr"] = item.get("group_bh_fdr", "")
            if _to_float(item.get("group_bh_fdr", 1.0)) <= 0.05:
                sig = True
        out["any_lr_pair_fdr_significant"] = sig
        out["fdr_status"] = "strict_candidate_fdr_supported" if row.candidate_status == "strict_candidate" and sig else "screened_or_not_strict"
        rows.append(out)
    return pd.DataFrame(rows)


def _audit_dataset(
    dataset_key: str,
    candidate_table: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    goal_cfg: dict[str, Any],
) -> dict[str, Any]:
    n_lr = len(lr_pairs)
    threshold = int(goal_cfg["strict_screen"]["baseline_rank_gt_or_absent"])
    strict_n = int((candidate_table["candidate_status"] == "strict_candidate").sum())
    if n_lr <= threshold:
        status = "blocked_by_lr_universe_size"
        reason = f"Only {n_lr} LR pairs are present; baseline rank >{threshold} cannot be evaluated."
    elif strict_n == 0:
        status = "no_strict_candidate"
        reason = "No LR passed all strict ranking, weak/context, and high-expression-control gates."
    else:
        status = "strict_candidate_found"
        reason = ""
    return {
        "dataset": dataset_key,
        "n_lr_pairs": n_lr,
        "baseline_rank_gt_threshold": threshold,
        "n_spatialself_top10": int((candidate_table["spatialself_rank"] <= 10).sum()),
        "n_strict_candidates": strict_n,
        "audit_status": status,
        "audit_reason": reason,
        "next_action": "build_expanded_tonsil_or_melanoma_lr_universe" if status == "blocked_by_lr_universe_size" else "",
    }


def _read_downstream(dataset: dict[str, str], *, method: str) -> dict[str, dict[str, Any]]:
    ddir = PROJECT_ROOT / dataset["downstream_dir"] / method
    paths = sorted(ddir.glob("*__downstream_target_correlation.csv"))
    if not paths:
        return {}
    frame = pd.read_csv(paths[0])
    return frame.set_index("lr_id").to_dict("index")


def _top_enrichment(top: pd.DataFrame, all_edges: pd.DataFrame, columns: str | list[str]) -> tuple[str, float]:
    if isinstance(columns, str):
        top_key = top[columns].astype(str)
        all_key = all_edges[columns].astype(str)
    else:
        top_key = top[columns].astype(str).agg("->".join, axis=1)
        all_key = all_edges[columns].astype(str).agg("->".join, axis=1)
    if top_key.empty:
        return "", np.nan
    best = top_key.value_counts().index[0]
    top_rate = float((top_key == best).mean())
    all_rate = float((all_key == best).mean())
    return best, top_rate / max(all_rate, 1e-12)


def _receiver_proxy_spearman(edges: pd.DataFrame, expression: pd.DataFrame, receptor: str) -> float | str:
    if receptor not in expression.columns or edges.empty:
        return ""
    incoming = edges.groupby("receiver_id")["score"].sum()
    expr = expression.set_index("cell_id")[receptor]
    common = incoming.index.intersection(expr.index)
    if len(common) < 3:
        return ""
    x = incoming.loc[common]
    y = expr.loc[common]
    if x.nunique() < 2 or y.nunique() < 2:
        return ""
    return float(x.corr(y, method="spearman"))


def _method_rank(group: pd.DataFrame, method: str) -> int | str:
    row = group[group["method"] == method]
    if row.empty or bool(row.iloc[0]["absent"]):
        return "absent"
    return int(row.iloc[0]["lr_rank"])


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
