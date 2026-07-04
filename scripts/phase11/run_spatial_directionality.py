#!/usr/bin/env python
"""COMMOT-style spatial signaling directionality for Visium HD CRC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOAL5_DIR = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"
DATASET_DIR = GOAL5_DIR / "standardized_inputs/visium_hd_crc_p2crc_subset"
OUT_DIR = PROJECT_ROOT / "outputs/phase11/commot_style_spatial_story/visium_hd_crc_p2crc_subset"

PREDICTIONS = {
    "spatialself": GOAL5_DIR
    / "visium_hd_crc_p2crc_spatialself_calibrated/spatialself_wavelet_edge_prior/"
    / "visium_hd_crc_p2crc_subset__spatialself_wavelet_predictions.csv",
    "commot": GOAL5_DIR
    / "visium_hd_crc_p2crc_baseline_panel/commot/"
    / "visium_hd_crc_p2crc_subset__commot_predictions.csv",
    "cellchat": GOAL5_DIR
    / "visium_hd_crc_p2crc_baseline_panel/cellchat/"
    / "visium_hd_crc_p2crc_subset__cellchat_predictions.csv",
    "cellnest": GOAL5_DIR
    / "visium_hd_crc_p2crc_baseline_panel/cellnest/"
    / "visium_hd_crc_p2crc_subset__cellnest_predictions.csv",
}

FOCUS_LR = {
    "SPP1-CD44": "visium_hd_crc_lr_000_SPP1_CD44",
    "CXCL12-CXCR4": "visium_hd_crc_lr_001_CXCL12_CXCR4",
    "VEGFA-KDR": "visium_hd_crc_lr_002_VEGFA_KDR",
    "TGFB1-TGFBR2": "visium_hd_crc_lr_003_TGFB1_TGFBR2",
    "CD274-PDCD1": "visium_hd_crc_lr_004_CD274_PDCD1",
    "ACTB-KRT8": "visium_hd_crc_lr_005_ACTB_KRT8_negative",
    "COL1A1-EPCAM": "visium_hd_crc_lr_006_COL1A1_EPCAM_negative",
}

EXPECTED_FLOW = {
    "SPP1-CD44": [
        ("tumor_myeloid_boundary", "tumor_stromal_boundary"),
        ("tumor_immune_boundary", "tumor_stromal_boundary"),
        ("tumor_myeloid_boundary", "tumor_immune_boundary"),
    ],
    "CXCL12-CXCR4": [
        ("tumor_stromal_boundary", "tumor_immune_boundary"),
        ("tumor_stromal_boundary", "tumor_core"),
    ],
    "VEGFA-KDR": [
        ("tumor_stromal_boundary", "vascular_stromal_boundary"),
        ("tumor_core", "vascular_stromal_boundary"),
    ],
    "TGFB1-TGFBR2": [
        ("tumor_core", "tumor_stromal_boundary"),
        ("tumor_immune_boundary", "tumor_stromal_boundary"),
    ],
    "CD274-PDCD1": [
        ("tumor_core", "tumor_immune_boundary"),
        ("tumor_immune_boundary", "tumor_immune_boundary"),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-edge-quantile", type=float, default=0.9)
    parser.add_argument("--min-score", type=float, default=0.0)
    args = parser.parse_args()

    result_dir = OUT_DIR / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    cells = pd.read_csv(DATASET_DIR / "cells.csv")
    candidates = pd.read_csv(DATASET_DIR / "candidate_interactions.csv")
    edge_tables = []
    receiver_vectors = []
    niche_flows = []
    lr_summaries = []
    for method, pred_path in PREDICTIONS.items():
        if not pred_path.exists():
            continue
        predictions = pd.read_csv(pred_path)
        edges = build_edge_table(candidates, predictions, cells, method)
        edge_tables.append(edges)
        receiver_vectors.append(build_receiver_vectors(edges))
        niche_flows.append(build_niche_flow(edges))
        lr_summaries.append(build_lr_summary(edges, args.top_edge_quantile, args.min_score))

    edge_all = pd.concat(edge_tables, ignore_index=True)
    receiver_all = pd.concat(receiver_vectors, ignore_index=True)
    niche_all = pd.concat(niche_flows, ignore_index=True)
    summary_all = pd.concat(lr_summaries, ignore_index=True)
    niche_all = annotate_expected_flow(niche_all)
    summary_all = summary_all.merge(expected_alignment(niche_all), on=["method", "lr_pair"], how="left")

    edge_all.to_csv(result_dir / "directionality_edges.csv", index=False)
    receiver_all.to_csv(result_dir / "directionality_receiver_vectors.csv", index=False)
    niche_all.to_csv(result_dir / "directionality_niche_flow.csv", index=False)
    summary_all.to_csv(result_dir / "directionality_lr_summary.csv", index=False)
    manifest = {
        "dataset": "visium_hd_crc_p2crc_subset",
        "methods": sorted(edge_all["method"].unique()),
        "focus_lr": FOCUS_LR,
        "expected_flow": {k: [list(x) for x in v] for k, v in EXPECTED_FLOW.items()},
        "outputs": [
            "directionality_edges.csv",
            "directionality_receiver_vectors.csv",
            "directionality_niche_flow.csv",
            "directionality_lr_summary.csv",
        ],
        "claim_boundary": (
            "Directionality is inferred from scored sender-to-receiver candidate edges and summarized as weighted spatial vectors. "
            "This is a COMMOT-style flow analysis, not time-resolved causal transport."
        ),
    }
    (result_dir / "directionality_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary_all.sort_values(["method", "lr_pair"]).to_string(index=False))


def build_edge_table(candidates: pd.DataFrame, predictions: pd.DataFrame, cells: pd.DataFrame, method: str) -> pd.DataFrame:
    cell_pos = cells[["cell_id", "x", "y", "cell_type", "niche"]].copy()
    send = cell_pos.rename(
        columns={
            "cell_id": "sender_id",
            "x": "sender_x",
            "y": "sender_y",
            "cell_type": "sender_cell_type_meta",
            "niche": "sender_niche_meta",
        }
    )
    recv = cell_pos.rename(
        columns={
            "cell_id": "receiver_id",
            "x": "receiver_x",
            "y": "receiver_y",
            "cell_type": "receiver_cell_type_meta",
            "niche": "receiver_niche_meta",
        }
    )
    edges = (
        candidates.merge(predictions[["candidate_id", "score"]], on="candidate_id", how="inner")
        .merge(send, on="sender_id", how="left")
        .merge(recv, on="receiver_id", how="left")
    )
    lr_lookup = {v: k for k, v in FOCUS_LR.items()}
    edges["lr_pair"] = edges["lr_id"].map(lr_lookup).fillna(edges["ligand"].astype(str) + "-" + edges["receptor"].astype(str))
    dx = edges["receiver_x"].to_numpy(dtype=float) - edges["sender_x"].to_numpy(dtype=float)
    dy = edges["receiver_y"].to_numpy(dtype=float) - edges["sender_y"].to_numpy(dtype=float)
    dist = np.sqrt(dx**2 + dy**2)
    edges["unit_dx"] = np.divide(dx, dist, out=np.zeros_like(dx), where=dist > 0)
    edges["unit_dy"] = np.divide(dy, dist, out=np.zeros_like(dy), where=dist > 0)
    edges["weighted_dx"] = edges["score"].to_numpy(dtype=float) * edges["unit_dx"]
    edges["weighted_dy"] = edges["score"].to_numpy(dtype=float) * edges["unit_dy"]
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
            "type_rule_satisfied",
            "niche_rule_satisfied",
            "false_positive_high_expression",
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
    )
    flow["flow_norm"] = np.sqrt(flow["mean_dx"] ** 2 + flow["mean_dy"] ** 2)
    flow["flow_label"] = flow["sender_niche"].astype(str) + " -> " + flow["receiver_niche"].astype(str)
    flow["rank_within_method_lr"] = flow.groupby(["method", "lr_pair"])["total_score"].rank(ascending=False, method="dense")
    return flow


def build_lr_summary(edges: pd.DataFrame, top_edge_quantile: float, min_score: float) -> pd.DataFrame:
    rows = []
    for (method, lr_pair), table in edges.groupby(["method", "lr_pair"], sort=False):
        score = table["score"].to_numpy(dtype=float)
        threshold = max(float(np.quantile(score, top_edge_quantile)), min_score) if len(score) else min_score
        top = table[table["score"].ge(threshold)].copy()
        weighted = np.maximum(table["score"].to_numpy(dtype=float), 0)
        total = float(np.sum(weighted))
        rows.append(
            {
                "method": method,
                "lr_pair": lr_pair,
                "n_edges": int(len(table)),
                "score_sum": float(table["score"].sum()),
                "score_mean": float(table["score"].mean()),
                "score_q90": float(np.quantile(score, 0.9)) if len(score) else np.nan,
                "top_edge_threshold": threshold,
                "n_top_edges": int(len(top)),
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
