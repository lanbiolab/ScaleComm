#!/usr/bin/env python
"""Run graphclust-aware Xenium Goal 5 validation.

This audit asks whether Xenium positive controls stay prioritized after
controlling for 10x gene-expression graph clusters, and whether high-expression
negative controls still leak through under the same structure-aware null.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOAL_DIR = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"
DATASET_DIR = GOAL_DIR / "standardized_inputs/xenium_human_lymph_node_subset"
GRAPHCLUST_PATH = GOAL_DIR / "xenium_analysis_zarr_subset_cell_group_assignments.csv"


PREDICTION_PATHS = {
    "spatialself_wavelet_edge_prior": GOAL_DIR
    / "xenium_human_lymph_node_spatialself/xenium_human_lymph_node_replicate_00__spatialself_wavelet_predictions.csv",
    "commot": GOAL_DIR / "xenium_human_lymph_node_commot/xenium_human_lymph_node_replicate_00__commot_predictions.csv",
    "cellchat": GOAL_DIR
    / "xenium_human_lymph_node_cellchat/cellchat/xenium_human_lymph_node_subset__cellchat_predictions.csv",
    "spatalk": GOAL_DIR
    / "xenium_human_lymph_node_spatalk/spatalk/xenium_human_lymph_node_subset__spatalk_predictions.csv",
    "cellnest": GOAL_DIR
    / "xenium_human_lymph_node_cellnest/cellnest/xenium_human_lymph_node_subset__cellnest_predictions.csv",
}


NULL_MODELS = ("graphclust_pair_permutation", "edge_lr_identity_permutation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-dir", default=str(GOAL_DIR))
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--graphclust-path", default=str(GRAPHCLUST_PATH))
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=202614)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--top-fraction", type=float, default=0.1)
    args = parser.parse_args()

    goal_dir = Path(args.goal_dir)
    dataset_dir = Path(args.dataset_dir)
    graphclust_path = Path(args.graphclust_path)
    output_dir = goal_dir / f"xenium_graphclust_validation_{int(args.n_permutations)}perm"
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(dataset_dir / "candidate_interactions.csv")
    lr_pairs = pd.read_csv(dataset_dir / "lr_pairs.csv")
    graphclust = pd.read_csv(graphclust_path)
    candidates = _attach_graphclust(candidates, graphclust)
    candidates.to_csv(output_dir / "xenium_graphclust_candidate_interactions.csv", index=False)

    rank_rows: list[pd.DataFrame] = []
    fdr_rows: list[pd.DataFrame] = []
    method_status_rows: list[dict] = []
    for method, prediction_path in PREDICTION_PATHS.items():
        if not prediction_path.exists():
            method_status_rows.append(
                {
                    "method": method,
                    "status": "missing_prediction",
                    "prediction_path": _rel(prediction_path),
                    "detail": "prediction CSV not found",
                }
            )
            continue
        predictions = pd.read_csv(prediction_path)
        if predictions.empty or "candidate_id" not in predictions or "score" not in predictions:
            method_status_rows.append(
                {
                    "method": method,
                    "status": "invalid_prediction",
                    "prediction_path": _rel(prediction_path),
                    "detail": "prediction CSV is empty or missing candidate_id/score",
                }
            )
            continue
        merged = _merge(candidates, predictions, method)
        rank_table = _rank_table(merged, lr_pairs, method)
        rank_table.to_csv(output_dir / f"{method}__graphclust_adjusted_lr_rank_table.csv", index=False)
        rank_rows.append(rank_table)

        for null_model in NULL_MODELS:
            fdr_table = _group_fdr(
                merged,
                group_columns=["lr_id", "ligand", "receptor", "mechanism"],
                null_model=null_model,
                n_permutations=int(args.n_permutations),
                random_seed=int(args.random_seed) + len(fdr_rows),
                top_fraction=float(args.top_fraction),
            ).merge(lr_pairs, on=["lr_id", "ligand", "receptor", "mechanism"], how="left")
            fdr_table["method"] = method
            fdr_table["null_model"] = null_model
            fdr_table.to_csv(output_dir / f"{method}__{null_model}__graphclust_lr_pair_fdr.csv", index=False)
            fdr_rows.append(fdr_table)

        method_status_rows.append(
            {
                "method": method,
                "status": "completed",
                "prediction_path": _rel(prediction_path),
                "detail": f"n_predictions={len(predictions)};n_merged={len(merged)}",
            }
        )

    ranks = pd.concat(rank_rows, ignore_index=True) if rank_rows else pd.DataFrame()
    fdr = pd.concat(fdr_rows, ignore_index=True) if fdr_rows else pd.DataFrame()
    control_summary = _control_summary(ranks, fdr, alpha=float(args.alpha))
    audit = _audit(control_summary, alpha=float(args.alpha))
    cluster_summary = _cluster_summary(candidates)
    method_status = pd.DataFrame(method_status_rows)

    ranks.to_csv(output_dir / "xenium_graphclust_adjusted_lr_rank_table.csv", index=False)
    fdr.to_csv(output_dir / "xenium_graphclust_lr_pair_fdr_all.csv", index=False)
    control_summary.to_csv(output_dir / "xenium_graphclust_control_summary.csv", index=False)
    audit.to_csv(output_dir / "xenium_graphclust_validation_audit.csv", index=False)
    cluster_summary.to_csv(output_dir / "xenium_graphclust_candidate_cluster_summary.csv", index=False)
    method_status.to_csv(output_dir / "xenium_graphclust_method_status.csv", index=False)

    manifest = {
        "dataset": "xenium_human_lymph_node_subset",
        "dataset_dir": _rel(dataset_dir),
        "graphclust_path": _rel(graphclust_path),
        "output_dir": _rel(output_dir),
        "n_permutations": int(args.n_permutations),
        "alpha": float(args.alpha),
        "top_fraction": float(args.top_fraction),
        "null_models": list(NULL_MODELS),
        "strict_graphclust_success": bool(
            (audit.get("graphclust_validation_status", pd.Series(dtype=str)) == "strict_graphclust_success").any()
        ),
        "claim_boundary": (
            "This is a supplemental structure-aware audit using 10x gene-expression graphclust assignments. "
            "It controls for broad expression-cluster-pair effects, but these clusters are unsupervised and are "
            "not curated cell-type or perturbation ground truth."
        ),
        "outputs": {
            "rank_table": _rel(output_dir / "xenium_graphclust_adjusted_lr_rank_table.csv"),
            "fdr_all": _rel(output_dir / "xenium_graphclust_lr_pair_fdr_all.csv"),
            "control_summary": _rel(output_dir / "xenium_graphclust_control_summary.csv"),
            "audit": _rel(output_dir / "xenium_graphclust_validation_audit.csv"),
            "cluster_summary": _rel(output_dir / "xenium_graphclust_candidate_cluster_summary.csv"),
            "method_status": _rel(output_dir / "xenium_graphclust_method_status.csv"),
        },
    }
    (output_dir / "xenium_graphclust_validation_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(control_summary.to_string(index=False))
    print(audit.to_string(index=False))
    print(json.dumps(manifest, indent=2))


def _attach_graphclust(candidates: pd.DataFrame, graphclust: pd.DataFrame) -> pd.DataFrame:
    mapping = graphclust[["cell_id", "cluster_id"]].copy()
    mapping["graphclust"] = mapping["cluster_id"].map(_cluster_label).fillna("unassigned")
    mapping = mapping[["cell_id", "graphclust"]].drop_duplicates("cell_id")
    out = candidates.merge(mapping.rename(columns={"cell_id": "sender_id", "graphclust": "sender_graphclust"}), on="sender_id", how="left")
    out = out.merge(mapping.rename(columns={"cell_id": "receiver_id", "graphclust": "receiver_graphclust"}), on="receiver_id", how="left")
    out["sender_graphclust"] = out["sender_graphclust"].fillna("unassigned")
    out["receiver_graphclust"] = out["receiver_graphclust"].fillna("unassigned")
    out["graphclust_pair"] = out["sender_graphclust"].astype(str) + "->" + out["receiver_graphclust"].astype(str)
    return out


def _cluster_label(value: object) -> str:
    if pd.isna(value):
        return "unassigned"
    return f"cluster_{int(float(value)):02d}"


def _merge(candidates: pd.DataFrame, predictions: pd.DataFrame, method: str) -> pd.DataFrame:
    pred = predictions[["candidate_id", "score"]].copy()
    pred["score"] = pd.to_numeric(pred["score"], errors="coerce").fillna(0.0)
    merged = candidates.merge(pred, on="candidate_id", how="inner", validate="one_to_one")
    merged["method"] = method
    pair_mean = merged.groupby("graphclust_pair", sort=False)["score"].transform("mean")
    pair_std = merged.groupby("graphclust_pair", sort=False)["score"].transform("std").replace(0, np.nan)
    merged["graphclust_residual_score"] = merged["score"] - pair_mean
    merged["graphclust_z_score"] = (merged["score"] - pair_mean) / pair_std
    merged["graphclust_z_score"] = merged["graphclust_z_score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    merged["graphclust_pair_score_percentile"] = merged.groupby("graphclust_pair", sort=False)["score"].rank(pct=True)
    return merged


def _rank_table(merged: pd.DataFrame, lr_pairs: pd.DataFrame, method: str) -> pd.DataFrame:
    rows = []
    for lr_id, group in merged.groupby("lr_id", sort=False):
        rows.append(
            {
                "method": method,
                "lr_id": lr_id,
                "raw_mean_score": float(group["score"].mean()),
                "raw_top_tail_mean": _top_tail_mean(group["score"].to_numpy(), 0.1),
                "graphclust_residual_mean": float(group["graphclust_residual_score"].mean()),
                "graphclust_residual_top_tail_mean": _top_tail_mean(group["graphclust_residual_score"].to_numpy(), 0.1),
                "graphclust_z_mean": float(group["graphclust_z_score"].mean()),
                "graphclust_percentile_mean": float(group["graphclust_pair_score_percentile"].mean()),
                "n_edges": int(len(group)),
                "n_graphclust_pairs": int(group["graphclust_pair"].nunique()),
            }
        )
    table = pd.DataFrame(rows).merge(lr_pairs, on="lr_id", how="left")
    table["raw_rank"] = table["raw_mean_score"].rank(ascending=False, method="min").astype(int)
    table["graphclust_residual_rank"] = table["graphclust_residual_mean"].rank(ascending=False, method="min").astype(int)
    table["graphclust_top_tail_rank"] = table["graphclust_residual_top_tail_mean"].rank(ascending=False, method="min").astype(int)
    table["graphclust_percentile_rank"] = table["graphclust_percentile_mean"].rank(ascending=False, method="min").astype(int)
    return table.sort_values(["method", "graphclust_top_tail_rank", "graphclust_residual_rank", "raw_rank"])


def _group_fdr(
    merged: pd.DataFrame,
    *,
    group_columns: list[str],
    null_model: str,
    n_permutations: int,
    random_seed: int,
    top_fraction: float,
) -> pd.DataFrame:
    group_indices, group_keys = _group_indices_with_keys(merged, group_columns)
    scores = merged["score"].to_numpy(dtype=float)
    observed = np.array([_top_tail_mean(scores[index], top_fraction) for index in group_indices], dtype=float)
    null_groups = _null_groups(merged, null_model)
    rng = np.random.default_rng(random_seed)
    exceed = np.zeros(len(group_indices), dtype=np.int32)
    null_sum = np.zeros(len(group_indices), dtype=float)
    null_sum_sq = np.zeros(len(group_indices), dtype=float)
    for _ in range(n_permutations):
        null_scores = _permute_by_group(scores, null_groups, rng)
        null_stats = np.array([_top_tail_mean(null_scores[index], top_fraction) for index in group_indices], dtype=float)
        exceed += null_stats >= observed
        null_sum += null_stats
        null_sum_sq += null_stats * null_stats
    p_values = (exceed + 1.0) / (n_permutations + 1.0)
    out = pd.DataFrame([{column: value for column, value in zip(group_columns, key)} for key in group_keys])
    out["group_statistic"] = observed
    out["group_p_value"] = p_values
    out["group_bh_fdr"] = _bh_adjust(p_values)
    out["group_significant"] = out["group_bh_fdr"] <= 0.05
    out["n_candidate_edges"] = [int(len(index)) for index in group_indices]
    out["null_mean"] = null_sum / max(n_permutations, 1)
    out["null_std"] = np.sqrt(np.maximum(null_sum_sq / max(n_permutations, 1) - out["null_mean"].to_numpy() ** 2, 0.0))
    out["top_fraction"] = float(top_fraction)
    out["n_permutations"] = int(n_permutations)
    return out.sort_values(["group_bh_fdr", "group_p_value", "group_statistic"], ascending=[True, True, False])


def _null_groups(table: pd.DataFrame, null_model: str) -> list[np.ndarray]:
    if null_model == "graphclust_pair_permutation":
        return _group_indices(table, ["graphclust_pair"])
    if null_model == "edge_lr_identity_permutation":
        return _group_indices(table, ["sender_id", "receiver_id"])
    raise ValueError(f"Unsupported null model: {null_model}")


def _group_indices(table: pd.DataFrame, columns: list[str]) -> list[np.ndarray]:
    return [
        group_index.to_numpy(dtype=int)
        for _, group_index in table.reset_index().groupby(columns, sort=False)["index"]
        if len(group_index) > 1
    ] or [np.arange(len(table))]


def _group_indices_with_keys(table: pd.DataFrame, columns: list[str]) -> tuple[list[np.ndarray], list[tuple]]:
    group_indices = []
    group_keys = []
    for key, group_index in table.reset_index().groupby(columns, dropna=False, sort=False)["index"]:
        if not isinstance(key, tuple):
            key = (key,)
        group_indices.append(group_index.to_numpy(dtype=int))
        group_keys.append(key)
    return group_indices, group_keys


def _permute_by_group(scores: np.ndarray, groups: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    out = scores.copy()
    for index in groups:
        if len(index) > 1:
            out[index] = rng.permutation(out[index])
    return out


def _top_tail_mean(values: np.ndarray | pd.Series, top_fraction: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    n_top = max(1, int(np.ceil(arr.size * top_fraction)))
    return float(np.sort(arr)[-n_top:].mean())


def _control_summary(ranks: pd.DataFrame, fdr: pd.DataFrame, *, alpha: float) -> pd.DataFrame:
    if ranks.empty:
        return pd.DataFrame()
    rows = []
    for method, group in ranks.groupby("method", sort=False):
        positives = group[~group["false_positive_high_expression"].fillna(False).astype(bool)]
        negatives = group[group["false_positive_high_expression"].fillna(False).astype(bool)]
        fdr_method = fdr[fdr["method"].astype(str).eq(str(method))] if not fdr.empty else pd.DataFrame()
        for rank_metric, rank_column in [
            ("raw_mean", "raw_rank"),
            ("graphclust_residual", "graphclust_residual_rank"),
            ("graphclust_top_tail", "graphclust_top_tail_rank"),
            ("graphclust_percentile", "graphclust_percentile_rank"),
        ]:
            best_pos = positives.sort_values(rank_column).iloc[0] if not positives.empty else pd.Series(dtype=object)
            best_neg = negatives.sort_values(rank_column).iloc[0] if not negatives.empty else pd.Series(dtype=object)
            rows.append(
                {
                    "method": method,
                    "summary_type": rank_metric,
                    "null_model": "",
                    "best_positive_control": best_pos.get("control_name", ""),
                    "best_positive_rank": best_pos.get(rank_column, ""),
                    "best_negative_control": best_neg.get("control_name", ""),
                    "best_negative_rank": best_neg.get(rank_column, ""),
                    "positive_top3": int((pd.to_numeric(positives[rank_column], errors="coerce") <= 3).sum()),
                    "negative_top3": int((pd.to_numeric(negatives[rank_column], errors="coerce") <= 3).sum()),
                    "n_positive_significant": "",
                    "n_negative_significant": "",
                    "best_positive_group_bh_fdr": "",
                    "best_negative_group_bh_fdr": "",
                    "control_status": _rank_status(best_pos.get(rank_column, 999), best_neg.get(rank_column, 999)),
                }
            )
        for null_model, fdr_group in fdr_method.groupby("null_model", sort=False):
            fdr_pos = fdr_group[~fdr_group["false_positive_high_expression"].fillna(False).astype(bool)]
            fdr_neg = fdr_group[fdr_group["false_positive_high_expression"].fillna(False).astype(bool)]
            rows.append(
                {
                    "method": method,
                    "summary_type": "group_fdr",
                    "null_model": null_model,
                    "best_positive_control": _best_control(fdr_pos),
                    "best_positive_rank": "",
                    "best_negative_control": _best_control(fdr_neg),
                    "best_negative_rank": "",
                    "positive_top3": "",
                    "negative_top3": "",
                    "n_positive_significant": int((pd.to_numeric(fdr_pos["group_bh_fdr"], errors="coerce") <= alpha).sum()),
                    "n_negative_significant": int((pd.to_numeric(fdr_neg["group_bh_fdr"], errors="coerce") <= alpha).sum()),
                    "best_positive_group_bh_fdr": _best_value(fdr_pos, "group_bh_fdr"),
                    "best_negative_group_bh_fdr": _best_value(fdr_neg, "group_bh_fdr"),
                    "control_status": _fdr_status(fdr_pos, fdr_neg, alpha=alpha),
                }
            )
    return pd.DataFrame(rows)


def _audit(control_summary: pd.DataFrame, *, alpha: float) -> pd.DataFrame:
    rows = []
    if control_summary.empty:
        return pd.DataFrame(rows)
    for method, group in control_summary.groupby("method", sort=False):
        top_tail = _match(group, "summary_type", "graphclust_top_tail")
        graph_fdr = _match2(group, "summary_type", "group_fdr", "null_model", "graphclust_pair_permutation")
        edge_fdr = _match2(group, "summary_type", "group_fdr", "null_model", "edge_lr_identity_permutation")
        positive_ranking = _to_float(top_tail.get("best_positive_rank", 999)) <= 3
        negative_ranking_leakage = _to_float(top_tail.get("best_negative_rank", 999)) <= 3
        graph_pos = _to_float(graph_fdr.get("n_positive_significant", 0)) > 0
        graph_neg = _to_float(graph_fdr.get("n_negative_significant", 0)) > 0
        edge_pos = _to_float(edge_fdr.get("n_positive_significant", 0)) > 0
        edge_neg = _to_float(edge_fdr.get("n_negative_significant", 0)) > 0
        if positive_ranking and not negative_ranking_leakage and graph_pos and not graph_neg and edge_pos and not edge_neg:
            status = "strict_graphclust_success"
        elif positive_ranking and not negative_ranking_leakage:
            status = "graphclust_ranking_support_only"
        elif graph_neg or edge_neg or negative_ranking_leakage:
            status = "graphclust_negative_control_leakage"
        else:
            status = "graphclust_no_positive_recovery"
        rows.append(
            {
                "method": method,
                "graphclust_validation_status": status,
                "graphclust_positive_ranking_pass": bool(positive_ranking),
                "graphclust_negative_ranking_leakage": bool(negative_ranking_leakage),
                "graphclust_pair_positive_fdr_recovery": bool(graph_pos),
                "graphclust_pair_negative_fdr_leakage": bool(graph_neg),
                "edge_lr_identity_positive_fdr_recovery": bool(edge_pos),
                "edge_lr_identity_negative_fdr_leakage": bool(edge_neg),
                "best_graphclust_positive": top_tail.get("best_positive_control", ""),
                "best_graphclust_positive_rank": top_tail.get("best_positive_rank", ""),
                "best_graphclust_negative": top_tail.get("best_negative_control", ""),
                "best_graphclust_negative_rank": top_tail.get("best_negative_rank", ""),
                "best_graphclust_positive_fdr": graph_fdr.get("best_positive_group_bh_fdr", ""),
                "best_graphclust_negative_fdr": graph_fdr.get("best_negative_group_bh_fdr", ""),
                "best_edge_positive_fdr": edge_fdr.get("best_positive_group_bh_fdr", ""),
                "best_edge_negative_fdr": edge_fdr.get("best_negative_group_bh_fdr", ""),
                "alpha": float(alpha),
            }
        )
    return pd.DataFrame(rows)


def _cluster_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    return (
        candidates.groupby(["graphclust_pair", "sender_graphclust", "receiver_graphclust"], dropna=False)
        .size()
        .reset_index(name="n_candidate_edges")
        .sort_values("n_candidate_edges", ascending=False)
    )


def _rank_status(best_positive_rank: object, best_negative_rank: object) -> str:
    pos = _to_float(best_positive_rank, 999)
    neg = _to_float(best_negative_rank, 999)
    if pos <= 3 and neg > 3:
        return "positive_top3_negative_suppressed"
    if neg <= pos:
        return "negative_control_ranks_as_high_or_higher"
    return "mixed_or_low_rank"


def _fdr_status(positives: pd.DataFrame, negatives: pd.DataFrame, *, alpha: float) -> str:
    pos = int((pd.to_numeric(positives.get("group_bh_fdr", pd.Series(dtype=float)), errors="coerce") <= alpha).sum())
    neg = int((pd.to_numeric(negatives.get("group_bh_fdr", pd.Series(dtype=float)), errors="coerce") <= alpha).sum())
    if pos > 0 and neg == 0:
        return "positive_significant_negative_suppressed"
    if neg > 0:
        return "negative_control_leakage"
    return "no_control_significant"


def _best_control(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    return str(frame.sort_values(["group_bh_fdr", "group_p_value", "group_statistic"], ascending=[True, True, False]).iloc[0].get("control_name", ""))


def _best_value(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return float("nan")
    return float(pd.to_numeric(frame[column], errors="coerce").min())


def _match(frame: pd.DataFrame, column: str, value: str) -> dict:
    if frame.empty or column not in frame:
        return {}
    rows = frame[frame[column].astype(str).eq(str(value))]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def _match2(frame: pd.DataFrame, column1: str, value1: str, column2: str, value2: str) -> dict:
    if frame.empty or column1 not in frame or column2 not in frame:
        return {}
    rows = frame[frame[column1].astype(str).eq(str(value1)) & frame[column2].astype(str).eq(str(value2))]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        if value == "":
            return default
        out = float(value)
        if np.isnan(out):
            return default
        return out
    except (TypeError, ValueError):
        return default


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    if p_values.size == 0:
        return p_values
    order = np.argsort(p_values)
    ranked = p_values[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
