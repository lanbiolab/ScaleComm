#!/usr/bin/env python
"""COMMOT-style lymph node signaling directionality for chemokine axes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "outputs/phase6/expanded_lr_universe/datasets/human_lymph_node_visium_phase6_expanded_replicate_00"
BASELINE_DIR = PROJECT_ROOT / "outputs/phase6/expanded_lr_universe/baselines/lymph_node_phase6_expanded"
OUT_DIR = PROJECT_ROOT / "outputs/phase11/commot_style_spatial_story/lymph_node_phase6_expanded"

PREDICTIONS = {
    "spatialself": BASELINE_DIR
    / "spatialself_wavelet_edge_prior/human_lymph_node_visium_phase6_expanded_replicate_00__spatialself_wavelet_predictions.csv",
    "commot": BASELINE_DIR / "commot/human_lymph_node_visium_phase6_expanded_replicate_00__commot_predictions.csv",
    "cellchat": BASELINE_DIR / "cellchat/human_lymph_node_visium_phase6_expanded_replicate_00__cellchat_predictions.csv",
    "cellnest": BASELINE_DIR / "cellnest/human_lymph_node_visium_phase6_expanded_replicate_00__cellnest_predictions.csv",
    "spatalk": BASELINE_DIR / "spatalk/human_lymph_node_visium_phase6_expanded_replicate_00__spatalk_predictions.csv",
}

FOCUS_LR = {
    "CCL19-CCR7": "CCL19_CCR7_Tzone_semisynthetic",
    "CCL21-CCR7": "CCL21_CCR7_Tzone_phase6_weak",
    "CXCL13-CXCR5": "CXCL13_CXCR5_GC_semisynthetic_weak",
    "CXCL12-CXCR4": "CXCL12_CXCR4_follicle_phase6_context",
    "CD48-CD244": "CD48_CD244_Tcell_phase6_contact_weak",
    "SPP1-CD44": "SPP1_CD44_boundary_semisynthetic",
    "ACTB-EGFR": "ACTB_EGFR_false_positive_phase6",
    "GAPDH-ITGB1": "GAPDH_ITGB1_false_positive_phase6",
}

EXPECTED_FLOW = {
    "CCL19-CCR7": [
        ("T cell zone", "T cell zone"),
        ("T cell zone", "Mixed"),
        ("Mixed", "T cell zone"),
    ],
    "CCL21-CCR7": [
        ("T cell zone", "T cell zone"),
        ("T cell zone", "Mixed"),
        ("Mixed", "T cell zone"),
    ],
    "CXCL13-CXCR5": [
        ("Germinal center", "Germinal center"),
        ("B follicle", "Germinal center"),
        ("Germinal center", "B follicle"),
        ("B follicle", "B follicle"),
    ],
    "CXCL12-CXCR4": [
        ("B follicle", "B follicle"),
        ("B follicle", "Germinal center"),
        ("Germinal center", "B follicle"),
    ],
    "CD48-CD244": [
        ("T cell zone", "T cell zone"),
        ("Mixed", "T cell zone"),
    ],
    "SPP1-CD44": [
        ("Mixed", "Mixed"),
        ("Mixed", "T cell zone"),
        ("T cell zone", "Mixed"),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-edge-quantile", type=float, default=0.9)
    args = parser.parse_args()

    result_dir = OUT_DIR / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    cells = pd.read_csv(DATASET_DIR / "cells.csv")
    candidates = pd.read_csv(DATASET_DIR / "candidate_interactions.csv")
    edge_tables = []
    vector_tables = []
    flow_tables = []
    summary_tables = []
    for method, pred_path in PREDICTIONS.items():
        if not pred_path.exists():
            continue
        predictions = pd.read_csv(pred_path)
        edges = build_edge_table(candidates, predictions, cells, method)
        edge_tables.append(edges)
        vector_tables.append(build_receiver_vectors(edges))
        flow_tables.append(build_niche_flow(edges))
        summary_tables.append(build_lr_summary(edges, args.top_edge_quantile))

    edge_all = pd.concat(edge_tables, ignore_index=True)
    vectors = pd.concat(vector_tables, ignore_index=True)
    flows = pd.concat(flow_tables, ignore_index=True)
    summary = pd.concat(summary_tables, ignore_index=True)
    flows = annotate_expected_flow(flows)
    summary = summary.merge(expected_alignment(flows), on=["method", "lr_pair"], how="left")

    edge_all.to_csv(result_dir / "lymph_node_directionality_edges.csv", index=False)
    vectors.to_csv(result_dir / "lymph_node_directionality_receiver_vectors.csv", index=False)
    flows.to_csv(result_dir / "lymph_node_directionality_niche_flow.csv", index=False)
    summary.to_csv(result_dir / "lymph_node_directionality_lr_summary.csv", index=False)
    manifest = {
        "dataset": "human_lymph_node_visium_phase6_expanded_replicate_00",
        "methods": sorted(edge_all["method"].unique()),
        "focus_lr": FOCUS_LR,
        "expected_flow": {key: [list(v) for v in values] for key, values in EXPECTED_FLOW.items()},
        "outputs": [
            "lymph_node_directionality_edges.csv",
            "lymph_node_directionality_receiver_vectors.csv",
            "lymph_node_directionality_niche_flow.csv",
            "lymph_node_directionality_lr_summary.csv",
        ],
        "claim_boundary": "Directionality is inferred from weighted sender-to-receiver candidate edges and summarized as spatial vectors/niche flows. It is not time-resolved causal transport.",
    }
    (result_dir / "lymph_node_directionality_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    focus = summary[summary["lr_pair"].isin(["CCL19-CCR7", "CCL21-CCR7", "CXCL13-CXCR5", "CXCL12-CXCR4", "CD48-CD244"])]
    print(focus.sort_values(["lr_pair", "method"]).to_string(index=False))


def build_edge_table(candidates: pd.DataFrame, predictions: pd.DataFrame, cells: pd.DataFrame, method: str) -> pd.DataFrame:
    cell_pos = cells[["cell_id", "x", "y", "cell_type", "niche", "region"]].copy()
    send = cell_pos.rename(
        columns={
            "cell_id": "sender_id",
            "x": "sender_x",
            "y": "sender_y",
            "cell_type": "sender_cell_type_meta",
            "niche": "sender_niche_meta",
            "region": "sender_region_meta",
        }
    )
    recv = cell_pos.rename(
        columns={
            "cell_id": "receiver_id",
            "x": "receiver_x",
            "y": "receiver_y",
            "cell_type": "receiver_cell_type_meta",
            "niche": "receiver_niche_meta",
            "region": "receiver_region_meta",
        }
    )
    edges = (
        candidates.merge(predictions[["candidate_id", "score"]], on="candidate_id", how="inner")
        .merge(send, on="sender_id", how="left")
        .merge(recv, on="receiver_id", how="left")
    )
    lr_lookup = {value: key for key, value in FOCUS_LR.items()}
    edges["lr_pair"] = edges["lr_id"].map(lr_lookup).fillna(edges["ligand"].astype(str) + "-" + edges["receptor"].astype(str))
    dx = edges["receiver_x"].to_numpy(dtype=float) - edges["sender_x"].to_numpy(dtype=float)
    dy = edges["receiver_y"].to_numpy(dtype=float) - edges["sender_y"].to_numpy(dtype=float)
    dist = np.sqrt(dx**2 + dy**2)
    edges["unit_dx"] = np.divide(dx, dist, out=np.zeros_like(dx), where=dist > 0)
    edges["unit_dy"] = np.divide(dy, dist, out=np.zeros_like(dy), where=dist > 0)
    score = pd.to_numeric(edges["score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    edges["weighted_dx"] = score * edges["unit_dx"]
    edges["weighted_dy"] = score * edges["unit_dy"]
    edges["method"] = method
    return edges[
        [
            "method",
            "candidate_id",
            "lr_pair",
            "lr_id",
            "sender_id",
            "receiver_id",
            "score",
            "distance",
            "sender_x",
            "sender_y",
            "receiver_x",
            "receiver_y",
            "unit_dx",
            "unit_dy",
            "weighted_dx",
            "weighted_dy",
            "sender_cell_type",
            "receiver_cell_type",
            "sender_niche",
            "receiver_niche",
            "ground_truth",
            "weak_signal",
            "false_positive_high_expression",
            "type_rule_satisfied",
            "niche_rule_satisfied",
        ]
    ]


def build_receiver_vectors(edges: pd.DataFrame) -> pd.DataFrame:
    grouped = edges.groupby(["method", "lr_pair", "receiver_id"], as_index=False).agg(
        incoming_score=("score", "mean"),
        vector_dx=("weighted_dx", "mean"),
        vector_dy=("weighted_dy", "mean"),
        n_edges=("candidate_id", "count"),
        receiver_x=("receiver_x", "first"),
        receiver_y=("receiver_y", "first"),
        receiver_cell_type=("receiver_cell_type", "first"),
        receiver_niche=("receiver_niche", "first"),
    )
    grouped["vector_norm"] = np.sqrt(grouped["vector_dx"] ** 2 + grouped["vector_dy"] ** 2)
    return grouped


def build_niche_flow(edges: pd.DataFrame) -> pd.DataFrame:
    flow = edges.groupby(["method", "lr_pair", "sender_niche", "receiver_niche"], as_index=False).agg(
        total_score=("score", "sum"),
        mean_score=("score", "mean"),
        n_edges=("candidate_id", "count"),
        mean_dx=("weighted_dx", "mean"),
        mean_dy=("weighted_dy", "mean"),
        median_distance=("distance", "median"),
        type_rule_fraction=("type_rule_satisfied", "mean"),
        niche_rule_fraction=("niche_rule_satisfied", "mean"),
        ground_truth_fraction=("ground_truth", "mean"),
    )
    flow["flow_norm"] = np.sqrt(flow["mean_dx"] ** 2 + flow["mean_dy"] ** 2)
    flow["flow_label"] = flow["sender_niche"].astype(str) + " -> " + flow["receiver_niche"].astype(str)
    flow["rank_within_method_lr"] = flow.groupby(["method", "lr_pair"])["total_score"].rank(ascending=False, method="dense")
    return flow


def build_lr_summary(edges: pd.DataFrame, top_edge_quantile: float) -> pd.DataFrame:
    rows = []
    for (method, lr_pair), table in edges.groupby(["method", "lr_pair"], sort=False):
        score = table["score"].to_numpy(dtype=float)
        threshold = float(np.nanquantile(score, top_edge_quantile)) if len(score) else np.nan
        top = table[table["score"].ge(threshold)].copy()
        weighted = np.maximum(score, 0.0)
        total = float(np.sum(weighted))
        rows.append(
            {
                "method": method,
                "lr_pair": lr_pair,
                "n_edges": int(len(table)),
                "score_sum": float(np.nansum(score)),
                "score_mean": float(np.nanmean(score)),
                "score_q90": float(np.nanquantile(score, 0.9)),
                "top_edge_threshold": threshold,
                "n_top_edges": int(len(top)),
                "top_ground_truth_fraction": float(top["ground_truth"].mean()) if len(top) else np.nan,
                "top_type_rule_fraction": float(top["type_rule_satisfied"].mean()) if len(top) else np.nan,
                "top_niche_rule_fraction": float(top["niche_rule_satisfied"].mean()) if len(top) else np.nan,
                "mean_vector_dx": float(np.average(table["unit_dx"], weights=weighted)) if total > 0 else 0.0,
                "mean_vector_dy": float(np.average(table["unit_dy"], weights=weighted)) if total > 0 else 0.0,
                "vector_resultant_length": float(
                    np.sqrt(
                        np.average(table["unit_dx"], weights=weighted) ** 2
                        + np.average(table["unit_dy"], weights=weighted) ** 2
                    )
                )
                if total > 0
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def annotate_expected_flow(niche_flow: pd.DataFrame) -> pd.DataFrame:
    expected = {
        (pair, sender, receiver)
        for pair, flows in EXPECTED_FLOW.items()
        for sender, receiver in flows
    }
    out = niche_flow.copy()
    out["expected_flow"] = [
        (row.lr_pair, row.sender_niche, row.receiver_niche) in expected for row in out.itertuples()
    ]
    return out


def expected_alignment(niche_flow: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, lr_pair), table in niche_flow.groupby(["method", "lr_pair"], sort=False):
        expected = table[table["expected_flow"]]
        total = float(table["total_score"].sum())
        expected_total = float(expected["total_score"].sum())
        best = table.sort_values("total_score", ascending=False).iloc[0]
        rows.append(
            {
                "method": method,
                "lr_pair": lr_pair,
                "expected_flow_score_fraction": expected_total / total if total > 0 else np.nan,
                "n_expected_flow_edges": int(expected["n_edges"].sum()) if not expected.empty else 0,
                "best_flow_label": best["flow_label"],
                "best_flow_score": float(best["total_score"]),
                "best_flow_is_expected": bool(best["expected_flow"]),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
