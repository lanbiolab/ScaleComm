#!/usr/bin/env python
"""Directionality and niche-flow analysis for native full-transcriptome lymph node Visium."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "outputs/phase11/commot_style_spatial_story/human_lymph_node_visium_full_subset"
DATASET_DIR = OUT_DIR / "dataset"
BASELINE_DIR = OUT_DIR / "baselines"
RESULT_DIR = OUT_DIR / "results"

PREDICTIONS = {
    "spatialself": BASELINE_DIR / "spatialself_wavelet_edge_prior/dataset__spatialself_wavelet_predictions.csv",
    "commot": BASELINE_DIR / "commot/dataset__commot_predictions.csv",
    "cellchat": BASELINE_DIR / "cellchat/dataset__cellchat_predictions.csv",
    "spatalk": BASELINE_DIR / "spatalk/dataset__spatalk_predictions.csv",
    "cellnest": BASELINE_DIR / "cellnest/dataset__cellnest_predictions.csv",
}

EXPECTED_FLOW = {
    "CCL19-CCR7": [
        ("T_cell_zone_proxy", "T_cell_zone_proxy"),
        ("stromal_follicle_proxy", "T_cell_zone_proxy"),
    ],
    "CCL21-CCR7": [
        ("T_cell_zone_proxy", "T_cell_zone_proxy"),
        ("stromal_follicle_proxy", "T_cell_zone_proxy"),
    ],
    "CXCL13-CXCR5": [
        ("B_follicle_proxy", "B_follicle_proxy"),
        ("germinal_center_proxy", "B_follicle_proxy"),
        ("B_follicle_proxy", "germinal_center_proxy"),
        ("germinal_center_proxy", "germinal_center_proxy"),
    ],
    "CXCL12-CXCR4": [
        ("stromal_follicle_proxy", "T_cell_zone_proxy"),
        ("stromal_follicle_proxy", "B_follicle_proxy"),
        ("stromal_follicle_proxy", "germinal_center_proxy"),
    ],
    "CD48-CD244": [
        ("T_cell_zone_proxy", "T_cell_zone_proxy"),
        ("B_follicle_proxy", "T_cell_zone_proxy"),
    ],
    "CD40LG-CD40": [
        ("T_cell_zone_proxy", "B_follicle_proxy"),
        ("T_cell_zone_proxy", "germinal_center_proxy"),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-edge-quantile", type=float, default=0.9)
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(DATASET_DIR / "cells.csv")
    candidates = pd.read_csv(DATASET_DIR / "candidate_interactions.csv")
    lr_pairs = pd.read_csv(DATASET_DIR / "lr_pairs.csv")
    lr_lookup = lr_pairs.set_index("lr_id")["control_name"].to_dict()

    edge_frames = []
    vector_frames = []
    flow_frames = []
    summary_frames = []
    status_rows = []
    for method, pred_path in PREDICTIONS.items():
        if not pred_path.exists():
            status_rows.append({"method": method, "status": "prediction_missing", "prediction_path": rel(pred_path)})
            continue
        predictions = pd.read_csv(pred_path)
        edges = build_edge_table(candidates, predictions, cells, method, lr_lookup)
        edge_frames.append(edges)
        vector_frames.append(build_receiver_vectors(edges))
        flows = annotate_expected_flow(build_niche_flow(edges))
        flow_frames.append(flows)
        summary = build_lr_summary(edges, flows, args.top_edge_quantile)
        summary_frames.append(summary)
        status_rows.append({"method": method, "status": "completed", "prediction_path": rel(pred_path)})

    edge_all = pd.concat(edge_frames, ignore_index=True) if edge_frames else pd.DataFrame()
    vectors = pd.concat(vector_frames, ignore_index=True) if vector_frames else pd.DataFrame()
    flows = pd.concat(flow_frames, ignore_index=True) if flow_frames else pd.DataFrame()
    summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()

    edge_all.to_csv(RESULT_DIR / "lymph_node_full_directionality_edges.csv", index=False)
    vectors.to_csv(RESULT_DIR / "lymph_node_full_directionality_receiver_vectors.csv", index=False)
    flows.to_csv(RESULT_DIR / "lymph_node_full_directionality_niche_flow.csv", index=False)
    summary.to_csv(RESULT_DIR / "lymph_node_full_directionality_lr_summary.csv", index=False)
    pd.DataFrame(status_rows).to_csv(RESULT_DIR / "lymph_node_full_directionality_status.csv", index=False)
    manifest = {
        "dataset": "human_lymph_node_visium_full_subset",
        "dataset_dir": rel(DATASET_DIR),
        "methods": sorted(PREDICTIONS),
        "expected_flow": {key: [list(item) for item in value] for key, value in EXPECTED_FLOW.items()},
        "top_edge_quantile": float(args.top_edge_quantile),
        "outputs": [
            "lymph_node_full_directionality_edges.csv",
            "lymph_node_full_directionality_receiver_vectors.csv",
            "lymph_node_full_directionality_niche_flow.csv",
            "lymph_node_full_directionality_lr_summary.csv",
        ],
        "claim_boundary": (
            "Directionality is inferred from weighted sender-to-receiver edges and proxy niches. "
            "It is not time-resolved causal transport, and proxy niches are marker-rule labels."
        ),
    }
    (RESULT_DIR / "lymph_node_full_directionality_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    focus = summary[summary["lr_pair"].isin(["CCL19-CCR7", "CCL21-CCR7", "CXCL13-CXCR5", "CXCL12-CXCR4"])]
    print(focus.sort_values(["lr_pair", "method"]).to_string(index=False))


def build_edge_table(
    candidates: pd.DataFrame,
    predictions: pd.DataFrame,
    cells: pd.DataFrame,
    method: str,
    lr_lookup: dict[str, str],
) -> pd.DataFrame:
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
    keep = [
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
        "weak_signal",
        "false_positive_high_expression",
        "type_rule_satisfied",
        "niche_rule_satisfied",
    ]
    return edges[keep]


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
        high_expression_negative=("false_positive_high_expression", "max"),
    )
    flow["flow_norm"] = np.sqrt(flow["mean_dx"] ** 2 + flow["mean_dy"] ** 2)
    flow["flow_label"] = flow["sender_niche"].astype(str) + " -> " + flow["receiver_niche"].astype(str)
    flow["rank_within_method_lr"] = flow.groupby(["method", "lr_pair"])["total_score"].rank(ascending=False, method="dense")
    return flow


def annotate_expected_flow(flow: pd.DataFrame) -> pd.DataFrame:
    expected = []
    for row in flow.itertuples(index=False):
        pairs = set(EXPECTED_FLOW.get(str(row.lr_pair), []))
        expected.append((str(row.sender_niche), str(row.receiver_niche)) in pairs)
    flow["expected_flow"] = expected
    return flow


def build_lr_summary(edges: pd.DataFrame, flows: pd.DataFrame, top_edge_quantile: float) -> pd.DataFrame:
    rows = []
    for (method, lr_pair), table in edges.groupby(["method", "lr_pair"], sort=False):
        score = table["score"].to_numpy(dtype=float)
        threshold = float(np.nanquantile(score, top_edge_quantile)) if len(score) else np.nan
        top = table[table["score"].ge(threshold)].copy()
        weighted = np.maximum(score, 0.0)
        total = float(np.sum(weighted))
        flow_sub = flows[(flows["method"].eq(method)) & (flows["lr_pair"].eq(lr_pair))]
        expected_score = float(flow_sub.loc[flow_sub["expected_flow"], "total_score"].sum()) if len(flow_sub) else np.nan
        total_flow_score = float(flow_sub["total_score"].sum()) if len(flow_sub) else np.nan
        best_flow = flow_sub.sort_values("total_score", ascending=False).head(1)
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
                "top_type_rule_fraction": float(top["type_rule_satisfied"].mean()) if len(top) else np.nan,
                "top_niche_rule_fraction": float(top["niche_rule_satisfied"].mean()) if len(top) else np.nan,
                "expected_flow_score_fraction": expected_score / total_flow_score if total_flow_score and total_flow_score > 0 else np.nan,
                "best_flow_label": best_flow.iloc[0]["flow_label"] if not best_flow.empty else "",
                "best_flow_total_score": float(best_flow.iloc[0]["total_score"]) if not best_flow.empty else np.nan,
                "best_flow_expected": bool(best_flow.iloc[0]["expected_flow"]) if not best_flow.empty else False,
                "high_expression_negative": bool(table["false_positive_high_expression"].max()),
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


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
