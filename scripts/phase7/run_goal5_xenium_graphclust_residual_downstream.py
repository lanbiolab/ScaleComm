#!/usr/bin/env python
"""Run receiver-graphclust residual downstream validation for Xenium."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOAL_DIR = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"
DATASET_DIR = GOAL_DIR / "standardized_inputs/xenium_human_lymph_node_subset"
GRAPHCLUST_PATH = GOAL_DIR / "xenium_analysis_zarr_subset_cell_group_assignments.csv"
TARGET_PATH = GOAL_DIR / "xenium_downstream_target_validation/xenium_curated_within_panel_target_signatures.csv"

PREDICTION_PATHS = {
    "spatialself_wavelet_edge_prior": GOAL_DIR
    / "xenium_human_lymph_node_spatialself/xenium_human_lymph_node_replicate_00__spatialself_wavelet_predictions.csv",
    "commot": GOAL_DIR / "xenium_human_lymph_node_commot/xenium_human_lymph_node_replicate_00__commot_predictions.csv",
    "cellchat": GOAL_DIR
    / "xenium_human_lymph_node_cellchat/cellchat/xenium_human_lymph_node_subset__cellchat_predictions.csv",
    "cellnest": GOAL_DIR
    / "xenium_human_lymph_node_cellnest/cellnest/xenium_human_lymph_node_subset__cellnest_predictions.csv",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-dir", default=str(GOAL_DIR))
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--graphclust-path", default=str(GRAPHCLUST_PATH))
    parser.add_argument("--target-path", default=str(TARGET_PATH))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=202615)
    parser.add_argument("--min-receivers", type=int, default=20)
    args = parser.parse_args()

    goal_dir = Path(args.goal_dir)
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir) if args.output_dir else goal_dir / f"xenium_graphclust_residual_downstream_{int(args.n_permutations)}perm"
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(dataset_dir / "candidate_interactions.csv")
    lr_pairs = pd.read_csv(dataset_dir / "lr_pairs.csv")
    expression = pd.read_csv(dataset_dir / "expression.csv")
    graphclust = pd.read_csv(args.graphclust_path)
    targets = pd.read_csv(args.target_path)

    receiver_table = _receiver_target_table(expression, graphclust, targets, lr_pairs)
    receiver_table.to_csv(output_dir / "xenium_receiver_graphclust_residual_target_activity.csv", index=False)

    rows = []
    status_rows = []
    for method, prediction_path in PREDICTION_PATHS.items():
        if not prediction_path.exists():
            status_rows.append({"method": method, "status": "missing_prediction", "prediction_path": _rel(prediction_path)})
            continue
        predictions = pd.read_csv(prediction_path)
        if predictions.empty:
            status_rows.append({"method": method, "status": "empty_prediction", "prediction_path": _rel(prediction_path)})
            continue
        method_rows = _method_correlations(
            candidates,
            predictions,
            receiver_table,
            method=method,
            n_permutations=int(args.n_permutations),
            random_seed=int(args.random_seed) + len(rows),
            min_receivers=int(args.min_receivers),
        )
        rows.extend(method_rows)
        status_rows.append(
            {
                "method": method,
                "status": "completed",
                "prediction_path": _rel(prediction_path),
                "n_rows": len(method_rows),
            }
        )

    correlation = pd.DataFrame(rows)
    if not correlation.empty:
        tested = correlation["status"].astype(str).eq("tested")
        correlation.loc[tested, "bh_fdr"] = _bh_adjust(correlation.loc[tested, "empirical_p_value"].to_numpy(dtype=float))
        correlation.loc[~tested, "bh_fdr"] = np.nan
    summary = _method_summary(correlation)
    claims = _claim_table(summary)
    status = pd.DataFrame(status_rows)

    correlation.to_csv(output_dir / "xenium_graphclust_residual_downstream_correlation.csv", index=False)
    summary.to_csv(output_dir / "xenium_graphclust_residual_downstream_method_summary.csv", index=False)
    claims.to_csv(output_dir / "xenium_graphclust_residual_downstream_claims.csv", index=False)
    status.to_csv(output_dir / "xenium_graphclust_residual_downstream_method_status.csv", index=False)

    manifest = {
        "dataset": "xenium_human_lymph_node_subset",
        "dataset_dir": _rel(dataset_dir),
        "graphclust_path": _rel(Path(args.graphclust_path)),
        "target_path": _rel(Path(args.target_path)),
        "output_dir": _rel(output_dir),
        "n_permutations": int(args.n_permutations),
        "min_receivers": int(args.min_receivers),
        "claim_boundary": (
            "Receiver target activity is residualized within 10x gene-expression graphclust clusters. "
            "This reduces broad cell-state leakage, but remains an observational target-response proxy in a targeted Xenium panel."
        ),
        "outputs": {
            "correlation": _rel(output_dir / "xenium_graphclust_residual_downstream_correlation.csv"),
            "method_summary": _rel(output_dir / "xenium_graphclust_residual_downstream_method_summary.csv"),
            "claims": _rel(output_dir / "xenium_graphclust_residual_downstream_claims.csv"),
            "receiver_activity": _rel(output_dir / "xenium_receiver_graphclust_residual_target_activity.csv"),
            "method_status": _rel(output_dir / "xenium_graphclust_residual_downstream_method_status.csv"),
        },
    }
    (output_dir / "xenium_graphclust_residual_downstream_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(claims.to_string(index=False))
    print(json.dumps(manifest, indent=2))


def _receiver_target_table(
    expression: pd.DataFrame,
    graphclust: pd.DataFrame,
    targets: pd.DataFrame,
    lr_pairs: pd.DataFrame,
) -> pd.DataFrame:
    expr = expression.copy()
    expr = expr.merge(
        graphclust[["cell_id", "cluster_id"]].drop_duplicates("cell_id"),
        on="cell_id",
        how="left",
    )
    expr["receiver_graphclust"] = expr["cluster_id"].map(_cluster_label).fillna("unassigned")
    gene_columns = set(expr.columns) - {"cell_id", "cluster_id", "receiver_graphclust"}
    rows = []
    target_map = {
        (str(row.ligand), str(row.receptor)): _split_genes(row.target_genes)
        for row in targets.itertuples(index=False)
    }
    for row in lr_pairs.itertuples(index=False):
        genes = [gene for gene in target_map.get((str(row.ligand), str(row.receptor)), []) if gene in gene_columns]
        if genes:
            activity = expr[genes].apply(pd.to_numeric, errors="coerce").fillna(0.0).mean(axis=1)
            residual = activity - activity.groupby(expr["receiver_graphclust"]).transform("mean")
            n_targets = len(genes)
            status = "usable"
        else:
            activity = pd.Series(np.nan, index=expr.index)
            residual = pd.Series(np.nan, index=expr.index)
            n_targets = 0
            status = "missing_targets"
        rows.append(
            pd.DataFrame(
                {
                    "receiver_id": expr["cell_id"],
                    "receiver_graphclust": expr["receiver_graphclust"],
                    "lr_id": row.lr_id,
                    "ligand": row.ligand,
                    "receptor": row.receptor,
                    "target_activity": activity,
                    "target_activity_graphclust_residual": residual,
                    "n_targets_present": n_targets,
                    "target_status": status,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _method_correlations(
    candidates: pd.DataFrame,
    predictions: pd.DataFrame,
    receiver_table: pd.DataFrame,
    *,
    method: str,
    n_permutations: int,
    random_seed: int,
    min_receivers: int,
) -> list[dict]:
    merged = candidates.merge(
        predictions[["candidate_id", "score"]],
        on="candidate_id",
        how="inner",
        validate="one_to_one",
    )
    merged["score"] = pd.to_numeric(merged["score"], errors="coerce").fillna(0.0)
    incoming = (
        merged.groupby(["lr_id", "receiver_id"], sort=False)["score"]
        .mean()
        .reset_index(name="incoming_score")
    )
    rng = np.random.default_rng(random_seed)
    rows = []
    for lr_id, group in incoming.groupby("lr_id", sort=False):
        target = receiver_table[receiver_table["lr_id"].astype(str).eq(str(lr_id))]
        test = group.merge(target, on=["lr_id", "receiver_id"], how="inner")
        test = test[test["target_status"].astype(str).eq("usable")].copy()
        test = test.dropna(subset=["incoming_score", "target_activity_graphclust_residual"])
        if len(test) < min_receivers or test["target_activity_graphclust_residual"].nunique() < 2:
            rows.append(_empty_row(method, lr_id, candidates, "too_few_receivers_or_constant_target", len(test)))
            continue
        rho, p_value = stats.spearmanr(test["incoming_score"], test["target_activity_graphclust_residual"])
        if not np.isfinite(rho):
            rows.append(_empty_row(method, lr_id, candidates, "nonfinite_correlation", len(test)))
            continue
        exceed = 0
        values = test["target_activity_graphclust_residual"].to_numpy(dtype=float)
        scores = test["incoming_score"].to_numpy(dtype=float)
        clusters = test["receiver_graphclust"].astype(str).to_numpy()
        for _ in range(n_permutations):
            null_values = _permute_within_labels(values, clusters, rng)
            null_rho, _ = stats.spearmanr(scores, null_values)
            if np.isfinite(null_rho) and null_rho >= rho:
                exceed += 1
        empirical = (exceed + 1.0) / (n_permutations + 1.0)
        meta = _lr_meta(candidates, lr_id)
        rows.append(
            {
                "method": method,
                "lr_id": lr_id,
                "ligand": meta.get("ligand", ""),
                "receptor": meta.get("receptor", ""),
                "mechanism": meta.get("mechanism", ""),
                "control_name": meta.get("control_name", ""),
                "false_positive_high_expression": bool(meta.get("false_positive_high_expression", False)),
                "n_receiver_cells": int(len(test)),
                "n_targets_present": int(test["n_targets_present"].iloc[0]),
                "spearman_r": float(rho),
                "spearman_p_value": float(p_value),
                "empirical_p_value": float(empirical),
                "status": "tested",
            }
        )
    return rows


def _empty_row(method: str, lr_id: str, candidates: pd.DataFrame, status: str, n_receiver_cells: int) -> dict:
    meta = _lr_meta(candidates, lr_id)
    return {
        "method": method,
        "lr_id": lr_id,
        "ligand": meta.get("ligand", ""),
        "receptor": meta.get("receptor", ""),
        "mechanism": meta.get("mechanism", ""),
        "control_name": meta.get("control_name", ""),
        "false_positive_high_expression": bool(meta.get("false_positive_high_expression", False)),
        "n_receiver_cells": int(n_receiver_cells),
        "n_targets_present": 0,
        "spearman_r": np.nan,
        "spearman_p_value": np.nan,
        "empirical_p_value": np.nan,
        "status": status,
    }


def _method_summary(correlation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if correlation.empty:
        return pd.DataFrame(rows)
    for method, group in correlation.groupby("method", sort=False):
        tested = group[group["status"].astype(str).eq("tested")].copy()
        positive = tested[~tested["false_positive_high_expression"].fillna(False).astype(bool)]
        negative = tested[tested["false_positive_high_expression"].fillna(False).astype(bool)]
        rows.append(
            {
                "method": method,
                "n_lr_pairs": int(group["lr_id"].nunique()),
                "n_tested": int(tested["lr_id"].nunique()),
                "best_positive_lr": _best_lr(positive),
                "best_positive_spearman": _best_value(positive, "spearman_r"),
                "best_positive_bh_fdr": _best_value(positive, "bh_fdr"),
                "best_negative_lr": _best_lr(negative),
                "best_negative_spearman": _best_value(negative, "spearman_r"),
                "best_negative_bh_fdr": _best_value(negative, "bh_fdr"),
                "n_positive_fdr_0_1": int(((positive["bh_fdr"] <= 0.1) & (positive["spearman_r"] > 0)).sum()) if len(positive) else 0,
                "n_negative_fdr_0_1": int(((negative["bh_fdr"] <= 0.1) & (negative["spearman_r"] > 0)).sum()) if len(negative) else 0,
                "mean_positive_spearman": _mean(positive["spearman_r"]),
                "mean_negative_spearman": _mean(negative["spearman_r"]),
                "positive_negative_delta": _mean(positive["spearman_r"]) - _mean(negative["spearman_r"]),
            }
        )
    return pd.DataFrame(rows)


def _claim_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in summary.itertuples(index=False):
        positive = int(row.n_positive_fdr_0_1) > 0
        negative = int(row.n_negative_fdr_0_1) > 0
        if positive and not negative:
            status = "graphclust_residual_functional_support_without_negative_fdr_signal"
        elif positive and negative:
            status = "mixed_residual_functional_support_negative_controls_also_correlate"
        else:
            status = "no_residual_functional_support"
        rows.append(
            {
                "method": row.method,
                "claim_status": status,
                "best_positive_lr": row.best_positive_lr,
                "best_negative_lr": row.best_negative_lr,
                "positive_negative_delta": row.positive_negative_delta,
                "allowed_claim": (
                    "Receiver target response remains observational and graphclust-residualized; it is stronger than "
                    "raw marker proxy correlation but not perturbation evidence."
                ),
            }
        )
    return pd.DataFrame(rows)


def _permute_within_labels(values: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = values.copy()
    for label in pd.unique(labels):
        idx = np.flatnonzero(labels == label)
        if len(idx) > 1:
            out[idx] = rng.permutation(out[idx])
    return out


def _lr_meta(candidates: pd.DataFrame, lr_id: str) -> dict:
    row = candidates[candidates["lr_id"].astype(str).eq(str(lr_id))]
    if row.empty:
        return {}
    meta = row.iloc[0].to_dict()
    return {
        "ligand": meta.get("ligand", ""),
        "receptor": meta.get("receptor", ""),
        "mechanism": meta.get("mechanism", ""),
        "control_name": f"{meta.get('ligand', '')}-{meta.get('receptor', '')}",
        "false_positive_high_expression": bool(meta.get("false_positive_high_expression", False)),
    }


def _cluster_label(value: object) -> str:
    if pd.isna(value):
        return "unassigned"
    return f"cluster_{int(float(value)):02d}"


def _split_genes(value: object) -> list[str]:
    return [gene.strip() for gene in str(value).replace(";", ",").split(",") if gene.strip()]


def _best_lr(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    return str(frame.sort_values(["spearman_r", "bh_fdr"], ascending=[False, True]).iloc[0]["lr_id"])


def _best_value(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return float("nan")
    return float(frame.sort_values(["spearman_r", "bh_fdr"], ascending=[False, True]).iloc[0][column])


def _mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else float("nan")


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
