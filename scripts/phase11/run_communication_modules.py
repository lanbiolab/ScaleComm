#!/usr/bin/env python
"""Cluster LR activity profiles into communication programs."""

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
PHASE10_DIR = PROJECT_ROOT / "outputs/phase10/full_transcriptome_commot_style/visium_hd_crc_p2crc_subset"
PHASE11_DIR = PROJECT_ROOT / "outputs/phase11/commot_style_spatial_story/visium_hd_crc_p2crc_subset"

PROGRAM_PRIORS = {
    "SPP1-CD44": "myeloid-remodeling",
    "TGFB1-TGFBR2": "ECM/stromal remodeling",
    "VEGFA-KDR": "angiogenesis",
    "CXCL12-CXCR4": "chemokine recruitment",
    "CD274-PDCD1": "immune checkpoint",
    "ACTB-KRT8": "high-expression negative",
    "COL1A1-EPCAM": "high-expression negative",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-modules", type=int, default=3)
    args = parser.parse_args()

    out = PHASE11_DIR / "results"
    out.mkdir(parents=True, exist_ok=True)
    incoming = pd.read_csv(PHASE10_DIR / "results/receiver_level_incoming_scores.csv")
    cells = pd.read_csv(
        PROJECT_ROOT
        / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/"
        / "standardized_inputs/visium_hd_crc_p2crc_subset/cells.csv"
    )
    pathway = pd.read_csv(PHASE10_DIR / "results/pathway_enrichment_summary.csv")
    lr_cols = [c for c in incoming.columns if c not in {"cell_id", "method"}]

    module_rows = []
    similarity_frames = []
    module_activity_frames = []
    enrichment_rows = []
    for method, table in incoming.groupby("method", sort=False):
        matrix = table.set_index("cell_id")[lr_cols].astype(float).T
        scaled = StandardScaler(with_mean=True, with_std=True).fit_transform(matrix)
        n_modules = min(args.n_modules, len(lr_cols) - 1)
        labels = AgglomerativeClustering(n_clusters=n_modules, linkage="ward").fit_predict(scaled) + 1
        sil = silhouette_score(scaled, labels) if len(set(labels)) > 1 and len(labels) > len(set(labels)) else np.nan
        for lr_pair, label in zip(matrix.index, labels):
            module_rows.append(
                {
                    "method": method,
                    "lr_pair": lr_pair,
                    "module": f"M{label}",
                    "program_prior": PROGRAM_PRIORS.get(lr_pair, "unknown"),
                    "silhouette_score_method": sil,
                }
            )
        similarity_frames.append(build_similarity(method, matrix))
        module_activity_frames.append(build_module_activity(method, table, cells, labels, lr_cols))
        enrichment_rows.extend(summarize_module_enrichment(method, labels, lr_cols, pathway))

    modules = pd.DataFrame(module_rows)
    similarity = pd.concat(similarity_frames, ignore_index=True)
    activity = pd.concat(module_activity_frames, ignore_index=True)
    enrichment = pd.DataFrame(enrichment_rows)
    module_summary = summarize_modules(modules, activity, enrichment)

    modules.to_csv(out / "communication_modules_lr_assignments.csv", index=False)
    similarity.to_csv(out / "communication_modules_lr_similarity.csv", index=False)
    activity.to_csv(out / "communication_modules_activity_by_niche.csv", index=False)
    enrichment.to_csv(out / "communication_modules_pathway_summary.csv", index=False)
    module_summary.to_csv(out / "communication_modules_summary.csv", index=False)
    manifest = {
        "dataset": "visium_hd_crc_p2crc_subset",
        "n_lr_pairs": len(lr_cols),
        "n_modules_requested": args.n_modules,
        "program_priors": PROGRAM_PRIORS,
        "outputs": [
            "communication_modules_lr_assignments.csv",
            "communication_modules_lr_similarity.csv",
            "communication_modules_activity_by_niche.csv",
            "communication_modules_pathway_summary.csv",
            "communication_modules_summary.csv",
        ],
        "claim_boundary": (
            "This is a small LR-panel module analysis for the current benchmark LR set. "
            "It supports communication-program framing but should be expanded to a larger LR database before strong module claims."
        ),
    }
    (out / "communication_modules_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(module_summary.to_string(index=False))


def build_similarity(method: str, matrix: pd.DataFrame) -> pd.DataFrame:
    corr = matrix.T.corr(method="spearman").fillna(0.0)
    rows = []
    if len(corr) > 1:
        distance = np.clip(1 - corr.to_numpy(), 0, 2)
        np.fill_diagonal(distance, 0)
        order = leaves_list(linkage(squareform(distance), method="average"))
    else:
        order = np.arange(len(corr))
    rank_map = {corr.index[i]: int(rank + 1) for rank, i in enumerate(order)}
    for a in corr.index:
        for b in corr.columns:
            rows.append(
                {
                    "method": method,
                    "lr_pair_a": a,
                    "lr_pair_b": b,
                    "spearman_similarity": float(corr.loc[a, b]),
                    "cluster_order_a": rank_map[a],
                    "cluster_order_b": rank_map[b],
                }
            )
    return pd.DataFrame(rows)


def build_module_activity(method: str, table: pd.DataFrame, cells: pd.DataFrame, labels: np.ndarray, lr_cols: list[str]) -> pd.DataFrame:
    assignments = pd.Series([f"M{x}" for x in labels], index=lr_cols)
    merged = table.merge(cells[["cell_id", "niche", "cell_type"]], on="cell_id", how="left")
    rows = []
    for module in sorted(assignments.unique()):
        cols = assignments[assignments.eq(module)].index.tolist()
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
                    "lr_pairs": ";".join(cols),
                }
            )
    return pd.DataFrame(rows)


def summarize_module_enrichment(method: str, labels: np.ndarray, lr_cols: list[str], pathway: pd.DataFrame) -> list[dict[str, object]]:
    assignments = pd.Series([f"M{x}" for x in labels], index=lr_cols)
    rows = []
    method_pathway = pathway[pathway["method"].eq(method)].copy()
    for module in sorted(assignments.unique()):
        pairs = assignments[assignments.eq(module)].index.tolist()
        sub = method_pathway[method_pathway["lr_pair"].isin(pairs)]
        if sub.empty:
            rows.append({"method": method, "module": module, "lr_pairs": ";".join(pairs), "status": "no_pathway_terms"})
            continue
        sig = sub[sub["n_significant_terms_fdr_0_05"].gt(0)]
        best = sub.sort_values(["best_term_fdr", "best_term_p"]).iloc[0]
        rows.append(
            {
                "method": method,
                "module": module,
                "lr_pairs": ";".join(pairs),
                "n_lrs": len(pairs),
                "n_lrs_with_significant_pathway_terms": int(sig["lr_pair"].nunique()),
                "total_significant_terms_fdr_0_05": int(sub["n_significant_terms_fdr_0_05"].sum()),
                "best_library": best["library"],
                "best_term": best["best_term"],
                "best_term_fdr": float(best["best_term_fdr"]),
                "status": "tested",
            }
        )
    return rows


def summarize_modules(modules: pd.DataFrame, activity: pd.DataFrame, enrichment: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, module), sub in modules.groupby(["method", "module"], sort=False):
        act = activity[(activity["method"].eq(method)) & (activity["module"].eq(module))]
        top_act = act.sort_values("mean_module_activity", ascending=False).head(1)
        enr = enrichment[(enrichment["method"].eq(method)) & (enrichment["module"].eq(module))]
        rows.append(
            {
                "method": method,
                "module": module,
                "lr_pairs": ";".join(sub["lr_pair"]),
                "program_priors": ";".join(sub["program_prior"]),
                "silhouette_score_method": sub["silhouette_score_method"].iloc[0],
                "top_niche": top_act.iloc[0]["niche"] if not top_act.empty else "",
                "top_cell_type": top_act.iloc[0]["cell_type"] if not top_act.empty else "",
                "top_niche_mean_activity": float(top_act.iloc[0]["mean_module_activity"]) if not top_act.empty else np.nan,
                "best_pathway_term": enr.iloc[0].get("best_term", "") if not enr.empty else "",
                "best_pathway_fdr": enr.iloc[0].get("best_term_fdr", np.nan) if not enr.empty else np.nan,
                "status": "small_lr_panel_module",
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
