#!/usr/bin/env python
"""Summarize COMMOT-style source-to-sink signaling trajectories across Phase 11 datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = PROJECT_ROOT / "outputs/phase11/commot_style_spatial_story"

DATASETS = {
    "visium_hd_crc": {
        "label": "Visium HD CRC",
        "root": BASE_DIR / "visium_hd_crc_p2crc_subset",
        "edge": "directionality_edges.csv",
        "flow": "directionality_niche_flow.csv",
        "lr_summary": "directionality_lr_summary.csv",
        "focus_lr": ["SPP1-CD44", "CXCL12-CXCR4", "VEGFA-KDR", "TGFB1-TGFBR2"],
    },
    "lymph_node_expanded": {
        "label": "Lymph node expanded benchmark",
        "root": BASE_DIR / "lymph_node_phase6_expanded",
        "edge": "lymph_node_directionality_edges.csv",
        "flow": "lymph_node_directionality_niche_flow.csv",
        "lr_summary": "lymph_node_directionality_lr_summary.csv",
        "focus_lr": ["CCL19-CCR7", "CCL21-CCR7", "CXCL13-CXCR5", "CXCL12-CXCR4", "CD48-CD244"],
    },
    "lymph_node_full": {
        "label": "Native human lymph node Visium",
        "root": BASE_DIR / "human_lymph_node_visium_full_subset",
        "edge": "lymph_node_full_directionality_edges.csv",
        "flow": "lymph_node_full_directionality_niche_flow.csv",
        "lr_summary": "lymph_node_full_directionality_lr_summary.csv",
        "focus_lr": ["CCL19-CCR7", "CCL21-CCR7", "CXCL13-CXCR5", "CD48-CD244"],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-quantile", type=float, default=0.9)
    parser.add_argument("--methods", nargs="+", default=["spatialself", "commot"])
    args = parser.parse_args()

    out_dir = BASE_DIR / "trajectory_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    flow_rows = []
    for key, cfg in DATASETS.items():
        result_dir = cfg["root"] / "results"
        edges_path = result_dir / cfg["edge"]
        flow_path = result_dir / cfg["flow"]
        lr_path = result_dir / cfg["lr_summary"]
        if not edges_path.exists() or not flow_path.exists() or not lr_path.exists():
            rows.append(
                {
                    "dataset_key": key,
                    "dataset": cfg["label"],
                    "status": "missing_directionality_inputs",
                    "missing": ";".join(str(p) for p in [edges_path, flow_path, lr_path] if not p.exists()),
                }
            )
            continue
        lr_summary = pd.read_csv(lr_path)
        flow = pd.read_csv(flow_path)
        focus = set(cfg["focus_lr"])
        usecols = [
            "method",
            "lr_pair",
            "score",
            "sender_x",
            "sender_y",
            "receiver_x",
            "receiver_y",
            "sender_niche",
            "receiver_niche",
            "distance",
            "type_rule_satisfied",
            "niche_rule_satisfied",
        ]
        chunks = pd.read_csv(edges_path, usecols=lambda c: c in usecols, chunksize=500_000)
        for chunk in chunks:
            chunk = chunk[chunk["method"].isin(args.methods) & chunk["lr_pair"].isin(focus)].copy()
            if chunk.empty:
                continue
            rows.extend(summarize_chunked_edges(key, cfg["label"], chunk, args.top_quantile))

        flow_focus = flow[flow["method"].isin(args.methods) & flow["lr_pair"].isin(focus)].copy()
        if not flow_focus.empty:
            flow_focus["dataset_key"] = key
            flow_focus["dataset"] = cfg["label"]
            flow_rows.append(flow_focus)

        if rows:
            # attach lr-level expected-flow metrics after chunk aggregation below
            pass

    edge_partial = pd.DataFrame(rows)
    if edge_partial.empty:
        raise SystemExit("No trajectory rows generated.")
    trajectory = finalize_edge_summaries(edge_partial)
    trajectory = attach_lr_summary(trajectory)
    flow_all = pd.concat(flow_rows, ignore_index=True) if flow_rows else pd.DataFrame()
    top_flow = summarize_top_flows(flow_all)
    merged = trajectory.merge(
        top_flow,
        on=["dataset_key", "dataset", "method", "lr_pair"],
        how="left",
    )
    merged.to_csv(out_dir / "phase11_signal_trajectory_summary.csv", index=False)
    if not flow_all.empty:
        flow_all.to_csv(out_dir / "phase11_signal_trajectory_niche_flow_all.csv", index=False)
    manifest = {
        "datasets": {key: cfg["label"] for key, cfg in DATASETS.items()},
        "methods": args.methods,
        "top_quantile": args.top_quantile,
        "trajectory_definition": (
            "Weighted centroid from sender coordinates to receiver coordinates among top-scoring LR edges; "
            "the vector is descriptive and not causal time-resolved transport."
        ),
        "outputs": [
            "phase11_signal_trajectory_summary.csv",
            "phase11_signal_trajectory_niche_flow_all.csv",
        ],
        "claim_boundary": (
            "Source-sink trajectories summarize scored spatial edges. They support directionality/niche-flow consistency "
            "but do not prove physical ligand transport or temporal causality."
        ),
    }
    (out_dir / "phase11_signal_trajectory_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(merged.sort_values(["dataset_key", "lr_pair", "method"]).to_string(index=False))


def summarize_chunked_edges(dataset_key: str, dataset_label: str, chunk: pd.DataFrame, top_quantile: float) -> list[dict[str, object]]:
    rows = []
    for (method, lr_pair), table in chunk.groupby(["method", "lr_pair"], sort=False):
        score = pd.to_numeric(table["score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        threshold = float(np.quantile(score, top_quantile)) if len(score) else np.nan
        top = table[pd.to_numeric(table["score"], errors="coerce").fillna(0.0).ge(threshold)].copy()
        rows.append(weighted_centroid_row(dataset_key, dataset_label, method, lr_pair, top, threshold, top_quantile))
    return rows


def weighted_centroid_row(
    dataset_key: str,
    dataset_label: str,
    method: str,
    lr_pair: str,
    table: pd.DataFrame,
    threshold: float,
    top_quantile: float,
) -> dict[str, object]:
    weights = pd.to_numeric(table["score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    weights = np.maximum(weights, 0.0)
    if weights.sum() <= 0:
        weights = np.ones(len(table), dtype=float)
    sx = weighted_mean(table["sender_x"], weights)
    sy = weighted_mean(table["sender_y"], weights)
    rx = weighted_mean(table["receiver_x"], weights)
    ry = weighted_mean(table["receiver_y"], weights)
    dx = rx - sx
    dy = ry - sy
    trajectory_length = float(np.sqrt(dx * dx + dy * dy))
    dominant_sender = top_category(table, "sender_niche", weights)
    dominant_receiver = top_category(table, "receiver_niche", weights)
    return {
        "dataset_key": dataset_key,
        "dataset": dataset_label,
        "method": method,
        "lr_pair": lr_pair,
        "status": "completed",
        "top_quantile": top_quantile,
        "edge_score_threshold": threshold,
        "n_top_edges": int(len(table)),
        "top_edge_score_sum": float(weights.sum()),
        "sender_centroid_x": sx,
        "sender_centroid_y": sy,
        "receiver_centroid_x": rx,
        "receiver_centroid_y": ry,
        "trajectory_dx": float(dx),
        "trajectory_dy": float(dy),
        "trajectory_length": trajectory_length,
        "median_top_edge_distance": float(pd.to_numeric(table["distance"], errors="coerce").median()) if "distance" in table else np.nan,
        "top_type_rule_fraction": float(pd.to_numeric(table["type_rule_satisfied"], errors="coerce").mean()) if "type_rule_satisfied" in table else np.nan,
        "top_niche_rule_fraction": float(pd.to_numeric(table["niche_rule_satisfied"], errors="coerce").mean()) if "niche_rule_satisfied" in table else np.nan,
        "dominant_sender_niche": dominant_sender[0],
        "dominant_sender_niche_score_fraction": dominant_sender[1],
        "dominant_receiver_niche": dominant_receiver[0],
        "dominant_receiver_niche_score_fraction": dominant_receiver[1],
        "centroid_flow_label": f"{dominant_sender[0]} -> {dominant_receiver[0]}",
    }


def finalize_edge_summaries(partial: pd.DataFrame) -> pd.DataFrame:
    # Rows can be split across chunks. Re-compute a weighted average of chunk-level centroids.
    rows = []
    for keys, table in partial.groupby(["dataset_key", "dataset", "method", "lr_pair"], sort=False):
        w = table["top_edge_score_sum"].to_numpy(dtype=float)
        if w.sum() <= 0:
            w = np.ones(len(table), dtype=float)
        sx = np.average(table["sender_centroid_x"], weights=w)
        sy = np.average(table["sender_centroid_y"], weights=w)
        rx = np.average(table["receiver_centroid_x"], weights=w)
        ry = np.average(table["receiver_centroid_y"], weights=w)
        dx = rx - sx
        dy = ry - sy
        first = table.iloc[0].to_dict()
        first.update(
            {
                "sender_centroid_x": float(sx),
                "sender_centroid_y": float(sy),
                "receiver_centroid_x": float(rx),
                "receiver_centroid_y": float(ry),
                "trajectory_dx": float(dx),
                "trajectory_dy": float(dy),
                "trajectory_length": float(np.sqrt(dx * dx + dy * dy)),
                "n_top_edges": int(table["n_top_edges"].sum()),
                "top_edge_score_sum": float(table["top_edge_score_sum"].sum()),
                "median_top_edge_distance": float(np.average(table["median_top_edge_distance"].fillna(0), weights=w)),
                "top_type_rule_fraction": float(np.average(table["top_type_rule_fraction"].fillna(0), weights=w)),
                "top_niche_rule_fraction": float(np.average(table["top_niche_rule_fraction"].fillna(0), weights=w)),
            }
        )
        rows.append(first)
    return pd.DataFrame(rows)


def attach_lr_summary(trajectory: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for key, cfg in DATASETS.items():
        path = cfg["root"] / "results" / cfg["lr_summary"]
        if not path.exists():
            continue
        lr = pd.read_csv(path)
        lr["dataset_key"] = key
        frames.append(lr)
    if not frames:
        return trajectory
    lr_all = pd.concat(frames, ignore_index=True)
    cols = [
        "dataset_key",
        "method",
        "lr_pair",
        "expected_flow_score_fraction",
        "best_flow_label",
        "best_flow_is_expected",
        "best_flow_expected",
        "score_mean",
        "score_sum",
    ]
    keep = [c for c in cols if c in lr_all.columns]
    lr_all = lr_all[keep].drop_duplicates(["dataset_key", "method", "lr_pair"])
    return trajectory.merge(lr_all, on=["dataset_key", "method", "lr_pair"], how="left")


def summarize_top_flows(flow_all: pd.DataFrame) -> pd.DataFrame:
    if flow_all.empty:
        return pd.DataFrame()
    rows = []
    for keys, table in flow_all.groupby(["dataset_key", "dataset", "method", "lr_pair"], sort=False):
        top = table.sort_values("total_score", ascending=False).head(1).iloc[0]
        expected = table[table.get("expected_flow", False).astype(bool)] if "expected_flow" in table else pd.DataFrame()
        rows.append(
            {
                "dataset_key": keys[0],
                "dataset": keys[1],
                "method": keys[2],
                "lr_pair": keys[3],
                "top_niche_flow_label": top["flow_label"],
                "top_niche_flow_score": float(top["total_score"]),
                "top_niche_flow_expected": bool(top["expected_flow"]) if "expected_flow" in top.index else False,
                "n_expected_niche_flows": int(len(expected)),
            }
        )
    return pd.DataFrame(rows)


def weighted_mean(values: pd.Series, weights: np.ndarray) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(arr) & np.isfinite(weights)
    if not ok.any():
        return np.nan
    return float(np.average(arr[ok], weights=weights[ok]))


def top_category(table: pd.DataFrame, column: str, weights: np.ndarray) -> tuple[str, float]:
    if column not in table:
        return "", np.nan
    tmp = pd.DataFrame({"category": table[column].astype(str).to_numpy(), "weight": weights})
    scores = tmp.groupby("category")["weight"].sum().sort_values(ascending=False)
    if scores.empty or scores.sum() <= 0:
        return "", np.nan
    return str(scores.index[0]), float(scores.iloc[0] / scores.sum())


if __name__ == "__main__":
    main()
