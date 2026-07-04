#!/usr/bin/env python
"""Cluster native lymph node full-transcriptome LR incoming profiles into communication modules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "outputs/phase11/commot_style_spatial_story/human_lymph_node_visium_full_subset"
DATASET_DIR = OUT_DIR / "dataset"
RESULT_DIR = OUT_DIR / "results"

PROGRAM_PRIOR = {
    "long_range_chemokine_niche": "chemokine recruitment",
    "juxtacrine_contact": "contact-dependent immune signaling",
    "negative_control_high_expression": "high-expression negative",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-modules", type=int, default=4)
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    incoming = pd.read_csv(RESULT_DIR / "lymph_node_full_receiver_level_incoming_scores.csv")
    cells = pd.read_csv(DATASET_DIR / "cells.csv")
    lr_pairs = pd.read_csv(DATASET_DIR / "lr_pairs.csv")
    impact = load_optional(RESULT_DIR / "lymph_node_full_lr_level_impact_summary.csv")
    direction = load_optional(RESULT_DIR / "lymph_node_full_directionality_lr_summary.csv")

    lr_cols = [column for column in incoming.columns if column not in {"cell_id", "method"}]
    lr_meta = lr_pairs.set_index("lr_id").reindex(lr_cols).reset_index()
    lr_meta["program_prior"] = lr_meta["mechanism"].map(PROGRAM_PRIOR).fillna("unknown")

    assignment_frames = []
    similarity_frames = []
    activity_frames = []
    enrichment_frames = []
    summary_frames = []
    for method, table in incoming.groupby("method", sort=False):
        matrix = table.set_index("cell_id")[lr_cols].astype(float).T
        scaled = StandardScaler(with_mean=True, with_std=True).fit_transform(matrix)
        n_modules = min(args.n_modules, max(2, len(lr_cols) - 1))
        labels = AgglomerativeClustering(n_clusters=n_modules, linkage="ward").fit_predict(scaled) + 1
        sil = silhouette_score(scaled, labels) if len(set(labels)) > 1 and len(labels) > len(set(labels)) else np.nan
        assignments = build_assignments(method, matrix.index.tolist(), labels, sil, lr_meta)
        assignment_frames.append(assignments)
        similarity_frames.append(build_similarity(method, matrix))
        activity = build_module_activity(method, table, cells, assignments)
        activity_frames.append(activity)
        enrichment = build_enrichment(method, assignments)
        enrichment_frames.append(enrichment)
        summary_frames.append(build_module_summary(method, assignments, activity, enrichment, impact, direction))

    assignments_all = pd.concat(assignment_frames, ignore_index=True)
    similarity_all = pd.concat(similarity_frames, ignore_index=True)
    activity_all = pd.concat(activity_frames, ignore_index=True)
    enrichment_all = pd.concat(enrichment_frames, ignore_index=True)
    summary_all = pd.concat(summary_frames, ignore_index=True)

    assignments_all.to_csv(RESULT_DIR / "lymph_node_full_modules_lr_assignments.csv", index=False)
    similarity_all.to_csv(RESULT_DIR / "lymph_node_full_modules_lr_similarity.csv", index=False)
    activity_all.to_csv(RESULT_DIR / "lymph_node_full_modules_activity_by_niche.csv", index=False)
    enrichment_all.to_csv(RESULT_DIR / "lymph_node_full_modules_mechanism_enrichment.csv", index=False)
    summary_all.to_csv(RESULT_DIR / "lymph_node_full_modules_summary.csv", index=False)
    manifest = {
        "dataset": "human_lymph_node_visium_full_subset",
        "n_lr_pairs": int(len(lr_cols)),
        "n_modules_requested": int(args.n_modules),
        "methods": sorted(incoming["method"].unique()),
        "program_prior_mapping": PROGRAM_PRIOR,
        "outputs": [
            "lymph_node_full_modules_lr_assignments.csv",
            "lymph_node_full_modules_lr_similarity.csv",
            "lymph_node_full_modules_activity_by_niche.csv",
            "lymph_node_full_modules_mechanism_enrichment.csv",
            "lymph_node_full_modules_summary.csv",
        ],
        "claim_boundary": (
            "Modules are clustered from receiver incoming profiles in a 9-LR native lymph-node first-pass panel. "
            "This supports communication-program framing but is not a complete LR atlas."
        ),
    }
    (RESULT_DIR / "lymph_node_full_modules_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary_all.to_string(index=False))


def load_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def build_assignments(method: str, lr_ids: list[str], labels: np.ndarray, sil: float, lr_meta: pd.DataFrame) -> pd.DataFrame:
    meta = lr_meta.set_index("lr_id")
    rows = []
    for lr_id, label in zip(lr_ids, labels):
        ann = meta.loc[lr_id].to_dict() if lr_id in meta.index else {}
        rows.append(
            {
                "method": method,
                "lr_id": lr_id,
                "lr_pair": ann.get("control_name", lr_id),
                "module": f"M{label}",
                "silhouette_score_method": sil,
                "ligand": ann.get("ligand", ""),
                "receptor": ann.get("receptor", ""),
                "mechanism": ann.get("mechanism", ""),
                "program_prior": ann.get("program_prior", "unknown"),
                "weak_signal": "weak" in str(ann.get("control_class", "")) or "context_specific" in str(ann.get("control_class", "")),
                "false_positive_high_expression": bool(ann.get("false_positive_high_expression", False)),
                "true_lr": bool(ann.get("true_lr", False)),
            }
        )
    return pd.DataFrame(rows)


def build_similarity(method: str, matrix: pd.DataFrame) -> pd.DataFrame:
    corr = matrix.T.corr(method="spearman").fillna(0.0)
    if len(corr) > 1:
        distance = np.clip(1 - corr.to_numpy(), 0, 2)
        np.fill_diagonal(distance, 0)
        order = leaves_list(linkage(squareform(distance), method="average"))
    else:
        order = np.arange(len(corr))
    rank = {corr.index[i]: int(j + 1) for j, i in enumerate(order)}
    rows = []
    for a in corr.index:
        for b in corr.columns:
            rows.append(
                {
                    "method": method,
                    "lr_pair_a": a,
                    "lr_pair_b": b,
                    "spearman_similarity": float(corr.loc[a, b]),
                    "cluster_order_a": rank[a],
                    "cluster_order_b": rank[b],
                }
            )
    return pd.DataFrame(rows)


def build_module_activity(method: str, table: pd.DataFrame, cells: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    merged = table.merge(cells[["cell_id", "niche", "cell_type", "region"]], on="cell_id", how="left")
    rows = []
    for module, sub_assign in assignments.groupby("module", sort=True):
        cols = sub_assign["lr_id"].tolist()
        merged[f"{module}_activity"] = merged[cols].mean(axis=1)
        for (niche, cell_type), sub in merged.groupby(["niche", "cell_type"], sort=False):
            rows.append(
                {
                    "method": method,
                    "module": module,
                    "niche": niche,
                    "cell_type": cell_type,
                    "n_cells": int(len(sub)),
                    "mean_module_activity": float(sub[f"{module}_activity"].mean()),
                    "q90_module_activity": float(sub[f"{module}_activity"].quantile(0.9)),
                    "lr_ids": ";".join(cols),
                }
            )
    return pd.DataFrame(rows)


def build_enrichment(method: str, assignments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    universe = assignments["program_prior"].value_counts()
    n_total = int(len(assignments))
    for module, sub in assignments.groupby("module", sort=True):
        n_module = int(len(sub))
        for prior, k in sub["program_prior"].value_counts().items():
            rows.append(
                {
                    "method": method,
                    "module": module,
                    "program_prior": prior,
                    "n_module_lr": n_module,
                    "n_prior_in_module": int(k),
                    "n_prior_total": int(universe.get(prior, 0)),
                    "module_fraction": float(k / n_module) if n_module else np.nan,
                    "universe_fraction": float(universe.get(prior, 0) / n_total) if n_total else np.nan,
                    "enrichment_ratio": float((k / n_module) / (universe.get(prior, 0) / n_total)) if n_module and universe.get(prior, 0) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_module_summary(
    method: str,
    assignments: pd.DataFrame,
    activity: pd.DataFrame,
    enrichment: pd.DataFrame,
    impact: pd.DataFrame,
    direction: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    impact_method = impact[impact["method"].eq(method)].copy() if not impact.empty else pd.DataFrame()
    direction_method = direction[direction["method"].eq(method)].copy() if not direction.empty else pd.DataFrame()
    for module, sub in assignments.groupby("module", sort=True):
        act = activity[(activity["method"].eq(method)) & (activity["module"].eq(module))]
        top = act.sort_values("mean_module_activity", ascending=False).head(1)
        enr = enrichment[(enrichment["method"].eq(method)) & (enrichment["module"].eq(module))].sort_values(
            ["enrichment_ratio", "module_fraction"], ascending=[False, False]
        )
        lr_ids = sub["lr_id"].tolist()
        lr_pairs = sub["lr_pair"].tolist()
        impact_sub = impact_method[impact_method["lr_id"].isin(lr_ids)] if not impact_method.empty else pd.DataFrame()
        direction_sub = direction_method[direction_method["lr_pair"].isin(lr_pairs)] if not direction_method.empty else pd.DataFrame()
        rows.append(
            {
                "method": method,
                "module": module,
                "n_lr": int(len(sub)),
                "lr_ids": ";".join(lr_ids),
                "lr_pairs": ";".join(lr_pairs),
                "program_priors": ";".join(sorted(sub["program_prior"].unique())),
                "n_true_lr": int(sub["true_lr"].sum()),
                "n_weak_lr": int(sub["weak_signal"].sum()),
                "n_high_expression_negative": int(sub["false_positive_high_expression"].sum()),
                "silhouette_score_method": float(sub["silhouette_score_method"].iloc[0]),
                "top_niche": top.iloc[0]["niche"] if not top.empty else "",
                "top_cell_type": top.iloc[0]["cell_type"] if not top.empty else "",
                "top_niche_mean_activity": float(top.iloc[0]["mean_module_activity"]) if not top.empty else np.nan,
                "top_enriched_program_prior": enr.iloc[0]["program_prior"] if not enr.empty else "",
                "top_program_enrichment_ratio": float(enr.iloc[0]["enrichment_ratio"]) if not enr.empty else np.nan,
                "mean_downstream_delta_r2": float(impact_sub["mean_delta_r2"].mean()) if not impact_sub.empty else np.nan,
                "n_impact_lr_fdr_0_1": int(impact_sub["n_impact_fdr_0_1"].gt(0).sum()) if not impact_sub.empty else 0,
                "mean_expected_flow_fraction": float(direction_sub["expected_flow_score_fraction"].mean()) if not direction_sub.empty else np.nan,
                "n_best_flows_expected": int(direction_sub["best_flow_expected"].astype(bool).sum()) if not direction_sub.empty else 0,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
