#!/usr/bin/env python
"""Stream high-resolution tonsil predictions into Goal 4 evidence tables.

This runner analyzes the existing high_spatial tonsil prediction artifact. It
does not retrain the Phase 2 default model; it is an auditable bridge for the
high-resolution real-data goal until a fresh wavelet_calibrated_fdr inference
run is available.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CONTROL_PRIORS = [
    {
        "control_name": "CXCL13-CXCR5",
        "control_class": "tonsil_positive_prior",
        "expected_senders": {"FDC", "T_follicular_helper"},
        "expected_receivers": {"B_germinal_center", "B_naive", "B_memory"},
    },
    {
        "control_name": "CXCL12-CXCR4",
        "control_class": "tonsil_positive_prior",
        "expected_senders": {"FDC", "myeloid", "mDC"},
        "expected_receivers": {"B_germinal_center", "B_naive", "B_memory", "T_CD4", "T_follicular_helper"},
    },
    {
        "control_name": "CD40-CD40LG",
        "control_class": "tonsil_positive_prior",
        "expected_senders": {"T_CD4", "T_follicular_helper"},
        "expected_receivers": {"B_germinal_center", "B_naive", "B_memory", "FDC"},
    },
    {
        "control_name": "ICOSLG-ICOS",
        "control_class": "tonsil_positive_prior",
        "expected_senders": {"B_germinal_center", "B_naive", "B_memory", "FDC", "mDC"},
        "expected_receivers": {"T_follicular_helper", "T_CD4"},
    },
    {
        "control_name": "TNFSF13B-TNFRSF13C",
        "control_class": "tonsil_positive_prior",
        "expected_senders": {"myeloid", "mDC", "FDC"},
        "expected_receivers": {"B_germinal_center", "B_naive", "B_memory", "plasma"},
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spatial", default="high_spatial/data/HumanTonsil_spatial.csv")
    parser.add_argument("--expression", default="high_spatial/data/HumanTonsil_expression.csv.gz")
    parser.add_argument("--graph", default="high_spatial/visium_lr_benchmark/graph_visium.pt")
    parser.add_argument("--lr-table", default="high_spatial/visium_lr_benchmark/lr_table.csv")
    parser.add_argument("--predictions", default="high_spatial/visium_lr_benchmark/pred_all_visium.csv")
    parser.add_argument("--output-dir", default="outputs/phase2/goal4_high_resolution/tonsil_existing_prediction_subset")
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--top-edges", type=int, default=50_000)
    parser.add_argument("--chunk-top-n", type=int, default=20_000)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap for smoke testing.")
    args = parser.parse_args()

    started = time.perf_counter()
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    spatial = _read_spatial(PROJECT_ROOT / args.spatial)
    lr_table = pd.read_csv(PROJECT_ROOT / args.lr_table)
    node_validation = _validate_node_order(spatial, PROJECT_ROOT / args.expression)
    graph_profile = _read_graph_profile(PROJECT_ROOT / args.graph)

    scan = _stream_predictions(
        PROJECT_ROOT / args.predictions,
        chunksize=args.chunksize,
        top_edges=args.top_edges,
        chunk_top_n=args.chunk_top_n,
        max_rows=args.max_rows,
    )
    annotated_top_edges = _annotate_top_edges(scan["top_edges"], spatial)
    scan["top_edges"] = annotated_top_edges
    lr_summary = _build_lr_summary(scan["lr_partials"], annotated_top_edges, spatial)
    cell_type_summary = _build_cell_type_pair_summary(annotated_top_edges)
    positive_summary = _build_positive_control_summary(lr_summary, annotated_top_edges)
    dataset_profile = _build_dataset_profile(
        args=args,
        spatial=spatial,
        lr_table=lr_table,
        graph_profile=graph_profile,
        node_validation=node_validation,
        scan=scan,
        elapsed_seconds=time.perf_counter() - started,
    )
    goal_summary = _build_goal_summary(dataset_profile, lr_summary, positive_summary, cell_type_summary)

    _write_csv(output_dir / "high_resolution_dataset_profile.csv", [dataset_profile])
    _write_csv(output_dir / "goal4_high_resolution_summary.csv", [goal_summary])
    lr_summary.to_csv(output_dir / "tonsil_lr_summary.csv", index=False)
    cell_type_summary.to_csv(output_dir / "tonsil_cell_type_pair_summary.csv", index=False)
    positive_summary.to_csv(output_dir / "tonsil_positive_control_summary.csv", index=False)
    scan["top_edges"].to_csv(output_dir / "tonsil_top_edges.csv", index=False)
    manifest = {
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "parameters": vars(args),
        "dataset_profile": dataset_profile,
        "goal_summary": goal_summary,
        "control_priors": [
            {
                "control_name": row["control_name"],
                "control_class": row["control_class"],
                "expected_senders": sorted(row["expected_senders"]),
                "expected_receivers": sorted(row["expected_receivers"]),
            }
            for row in CONTROL_PRIORS
        ],
    }
    (output_dir / "goal4_high_resolution_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Goal 4 dataset profile: {output_dir / 'high_resolution_dataset_profile.csv'}")
    print(f"Goal 4 summary:         {output_dir / 'goal4_high_resolution_summary.csv'}")
    print(f"LR summary:             {output_dir / 'tonsil_lr_summary.csv'}")
    print(f"Positive controls:      {output_dir / 'tonsil_positive_control_summary.csv'}")


def _read_spatial(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not df.empty and str(df.iloc[0]["NAME"]) == "TYPE":
        df = df.iloc[1:].copy()
    df = df.rename(columns={"NAME": "cell_id", "X": "x", "Y": "y"})
    df["x"] = pd.to_numeric(df["x"])
    df["y"] = pd.to_numeric(df["y"])
    df["node_id"] = np.arange(len(df), dtype=np.int64)
    df["cell_type"] = df["cell_type"].astype(str)
    return df.reset_index(drop=True)


def _validate_node_order(spatial: pd.DataFrame, expression_path: Path) -> dict[str, object]:
    result = {
        "expression_header_checked": False,
        "n_expression_cells": math.nan,
        "first_10_match": False,
        "all_header_cells_match_spatial": False,
    }
    if not expression_path.exists():
        return result
    header = pd.read_csv(expression_path, nrows=0).columns.tolist()[1:]
    spatial_ids = spatial["cell_id"].tolist()
    result["expression_header_checked"] = True
    result["n_expression_cells"] = len(header)
    result["first_10_match"] = header[:10] == spatial_ids[:10]
    result["all_header_cells_match_spatial"] = header == spatial_ids
    return result


def _read_graph_profile(path: Path) -> dict[str, object]:
    result = {"graph_checked": False, "graph_nodes": math.nan, "graph_edges": math.nan}
    if not path.exists():
        return result
    try:
        import torch

        graph = torch.load(path, map_location="cpu")
        result["graph_checked"] = True
        result["graph_nodes"] = int(graph.x.shape[0]) if hasattr(graph, "x") else math.nan
        result["graph_edges"] = int(graph.edge_index.shape[1]) if hasattr(graph, "edge_index") else math.nan
    except Exception as exc:  # pragma: no cover - defensive artifact reader
        result["graph_error"] = repr(exc)
    return result


def _stream_predictions(
    path: Path,
    *,
    chunksize: int,
    top_edges: int,
    chunk_top_n: int,
    max_rows: int | None,
) -> dict[str, object]:
    usecols = ["src", "dst", "lr_id", "ligand", "receptor", "dist", "prob"]
    dtypes = {
        "src": "int32",
        "dst": "int32",
        "lr_id": "int32",
        "ligand": "category",
        "receptor": "category",
        "dist": "float32",
        "prob": "float32",
    }
    lr_partials: list[pd.DataFrame] = []
    top_parts: list[pd.DataFrame] = []
    rows_seen = 0
    chunks_seen = 0
    for chunk in pd.read_csv(path, usecols=usecols, dtype=dtypes, chunksize=chunksize):
        if max_rows is not None:
            remaining = max_rows - rows_seen
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining].copy()
        chunks_seen += 1
        rows_seen += len(chunk)
        grouped = (
            chunk.groupby(["lr_id", "ligand", "receptor"], observed=True)
            .agg(
                n_candidate_edges=("prob", "size"),
                prob_sum=("prob", "sum"),
                prob_max=("prob", "max"),
                dist_sum=("dist", "sum"),
                dist_min=("dist", "min"),
                dist_max=("dist", "max"),
            )
            .reset_index()
        )
        lr_partials.append(grouped)
        top_parts.append(chunk.nlargest(min(chunk_top_n, len(chunk)), "prob").copy())
    if not lr_partials:
        raise RuntimeError(f"No prediction rows read from {path}")
    top_pool = pd.concat(top_parts, ignore_index=True)
    top_df = top_pool.nlargest(min(top_edges, len(top_pool)), "prob").copy()
    return {
        "rows_seen": rows_seen,
        "chunks_seen": chunks_seen,
        "lr_partials": lr_partials,
        "top_pool_rows": len(top_pool),
        "top_edges": top_df.reset_index(drop=True),
    }


def _build_lr_summary(lr_partials: Iterable[pd.DataFrame], top_edges: pd.DataFrame, spatial: pd.DataFrame) -> pd.DataFrame:
    partials = pd.concat(list(lr_partials), ignore_index=True)
    summary = (
        partials.groupby(["lr_id", "ligand", "receptor"], observed=True)
        .agg(
            n_candidate_edges=("n_candidate_edges", "sum"),
            prob_sum=("prob_sum", "sum"),
            prob_max=("prob_max", "max"),
            dist_sum=("dist_sum", "sum"),
            dist_min=("dist_min", "min"),
            dist_max=("dist_max", "max"),
        )
        .reset_index()
    )
    summary["prob_mean"] = summary["prob_sum"] / summary["n_candidate_edges"]
    summary["dist_mean"] = summary["dist_sum"] / summary["n_candidate_edges"]
    summary["canonical_pair"] = [_canonical_pair(lig, rec) for lig, rec in zip(summary["ligand"], summary["receptor"])]
    summary = summary.drop(columns=["prob_sum", "dist_sum"])

    annotated_top = top_edges
    pair_key = annotated_top["sender_cell_type"].astype(str) + "->" + annotated_top["receiver_cell_type"].astype(str)
    annotated_top = annotated_top.assign(cell_type_pair=pair_key)
    all_dispersion = _coordinate_dispersion(spatial["x"].to_numpy(), spatial["y"].to_numpy())
    top_rows = []
    for lr_id, group in annotated_top.groupby("lr_id", observed=True):
        pair_counts = group["cell_type_pair"].value_counts()
        dominant_pair = str(pair_counts.index[0]) if len(pair_counts) else ""
        receiver_dispersion = _coordinate_dispersion(group["receiver_x"].to_numpy(), group["receiver_y"].to_numpy())
        top_rows.append(
            {
                "lr_id": int(lr_id),
                "top_edge_count": int(len(group)),
                "top_prob_mean": float(group["prob"].mean()),
                "top_prob_max": float(group["prob"].max()),
                "top_dist_mean": float(group["dist"].mean()),
                "top_dist_median": float(group["dist"].median()),
                "dominant_cell_type_pair": dominant_pair,
                "dominant_cell_type_pair_fraction": float(pair_counts.iloc[0] / len(group)) if len(pair_counts) else 0.0,
                "cell_type_specificity": _specificity(pair_counts.to_numpy(dtype=float)),
                "receiver_spatial_dispersion": receiver_dispersion,
                "receiver_spatial_concentration": float(1.0 - receiver_dispersion / all_dispersion) if all_dispersion > 0 else math.nan,
            }
        )
    top_summary = pd.DataFrame(top_rows)
    if top_summary.empty:
        summary["top_edge_count"] = 0
    else:
        summary = summary.merge(top_summary, on="lr_id", how="left")
    for column in [
        "top_edge_count",
        "top_prob_mean",
        "top_prob_max",
        "top_dist_mean",
        "top_dist_median",
        "dominant_cell_type_pair_fraction",
        "cell_type_specificity",
        "receiver_spatial_dispersion",
        "receiver_spatial_concentration",
    ]:
        if column in summary:
            summary[column] = summary[column].fillna(0.0)
    if "dominant_cell_type_pair" in summary:
        summary["dominant_cell_type_pair"] = summary["dominant_cell_type_pair"].fillna("")
    summary["rank_by_prob_mean"] = summary["prob_mean"].rank(method="min", ascending=False).astype(int)
    summary["rank_by_prob_max"] = summary["prob_max"].rank(method="min", ascending=False).astype(int)
    summary["rank_by_top_edge_count"] = summary["top_edge_count"].rank(method="min", ascending=False).astype(int)
    return summary.sort_values(["rank_by_prob_mean", "lr_id"]).reset_index(drop=True)


def _build_cell_type_pair_summary(top_edges: pd.DataFrame) -> pd.DataFrame:
    if top_edges.empty:
        return pd.DataFrame()
    grouped = (
        top_edges.groupby(["sender_cell_type", "receiver_cell_type"], observed=True)
        .agg(
            n_top_edges=("prob", "size"),
            mean_prob=("prob", "mean"),
            max_prob=("prob", "max"),
            median_dist=("dist", "median"),
            n_lr_pairs=("lr_id", "nunique"),
        )
        .reset_index()
    )
    total = grouped["n_top_edges"].sum()
    grouped["top_edge_fraction"] = grouped["n_top_edges"] / total if total else 0.0
    return grouped.sort_values(["n_top_edges", "mean_prob"], ascending=[False, False]).reset_index(drop=True)


def _build_positive_control_summary(lr_summary: pd.DataFrame, top_edges: pd.DataFrame) -> pd.DataFrame:
    rows = []
    top_edges = top_edges.copy()
    if not top_edges.empty:
        top_edges["canonical_pair"] = [_canonical_pair(lig, rec) for lig, rec in zip(top_edges["ligand"], top_edges["receptor"])]
    n_lr = len(lr_summary)
    for prior in CONTROL_PRIORS:
        canonical = _canonical_pair(*prior["control_name"].split("-"))
        matches = lr_summary[lr_summary["canonical_pair"] == canonical]
        present = not matches.empty
        best = matches.sort_values("rank_by_prob_mean").iloc[0].to_dict() if present else {}
        top_match = top_edges[top_edges["canonical_pair"] == canonical] if present and not top_edges.empty else pd.DataFrame()
        directional_fraction = _expected_fraction(
            top_match,
            prior["expected_senders"],
            prior["expected_receivers"],
            allow_reverse=False,
        )
        unordered_fraction = _expected_fraction(
            top_match,
            prior["expected_senders"],
            prior["expected_receivers"],
            allow_reverse=True,
        )
        rows.append(
            {
                "control_name": prior["control_name"],
                "control_class": prior["control_class"],
                "canonical_pair": canonical,
                "present_in_lr_table": bool(present),
                "lr_id": int(best.get("lr_id", -1)) if present else -1,
                "ligand": best.get("ligand", ""),
                "receptor": best.get("receptor", ""),
                "rank_by_prob_mean": int(best.get("rank_by_prob_mean", 0)) if present else 0,
                "rank_percentile_by_prob_mean": float(best.get("rank_by_prob_mean", math.nan)) / n_lr if present and n_lr else math.nan,
                "rank_by_top_edge_count": int(best.get("rank_by_top_edge_count", 0)) if present else 0,
                "prob_mean": float(best.get("prob_mean", math.nan)) if present else math.nan,
                "prob_max": float(best.get("prob_max", math.nan)) if present else math.nan,
                "top_edge_count": int(best.get("top_edge_count", 0)) if present else 0,
                "dominant_cell_type_pair": best.get("dominant_cell_type_pair", ""),
                "cell_type_specificity": float(best.get("cell_type_specificity", math.nan)) if present else math.nan,
                "receiver_spatial_concentration": float(best.get("receiver_spatial_concentration", math.nan)) if present else math.nan,
                "directional_expected_cell_type_fraction": directional_fraction,
                "unordered_expected_cell_type_fraction": unordered_fraction,
            }
        )
    return pd.DataFrame(rows)


def _annotate_top_edges(top_edges: pd.DataFrame, spatial: pd.DataFrame) -> pd.DataFrame:
    result = top_edges.copy()
    cell_types = spatial["cell_type"].to_numpy()
    xs = spatial["x"].to_numpy(dtype=float)
    ys = spatial["y"].to_numpy(dtype=float)
    src = result["src"].to_numpy(dtype=int)
    dst = result["dst"].to_numpy(dtype=int)
    if src.min() < 0 or dst.min() < 0 or src.max() >= len(spatial) or dst.max() >= len(spatial):
        raise ValueError("Prediction src/dst indices are outside the spatial table node range.")
    result["sender_cell_type"] = cell_types[src]
    result["receiver_cell_type"] = cell_types[dst]
    result["sender_x"] = xs[src]
    result["sender_y"] = ys[src]
    result["receiver_x"] = xs[dst]
    result["receiver_y"] = ys[dst]
    return result


def _build_dataset_profile(
    *,
    args: argparse.Namespace,
    spatial: pd.DataFrame,
    lr_table: pd.DataFrame,
    graph_profile: dict[str, object],
    node_validation: dict[str, object],
    scan: dict[str, object],
    elapsed_seconds: float,
) -> dict[str, object]:
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    if platform.system() == "Darwin":
        peak_mb /= 1024.0
    return {
        "dataset": "human_tonsil_slide_tags",
        "analysis_label": "existing_prediction_stream",
        "is_fresh_phase2_default_inference": False,
        "n_cells": int(len(spatial)),
        "n_cell_types": int(spatial["cell_type"].nunique()),
        "n_lr_pairs": int(len(lr_table)),
        "n_prediction_rows_processed": int(scan["rows_seen"]),
        "n_prediction_chunks": int(scan["chunks_seen"]),
        "n_top_pool_rows": int(scan["top_pool_rows"]),
        "n_top_edges_output": int(len(scan["top_edges"])),
        "chunksize": int(args.chunksize),
        "top_edges_requested": int(args.top_edges),
        "chunk_top_n": int(args.chunk_top_n),
        "max_rows": args.max_rows if args.max_rows is not None else "",
        "wall_clock_seconds": elapsed_seconds,
        "peak_rss_mb": peak_mb,
        **graph_profile,
        **node_validation,
    }


def _build_goal_summary(
    dataset_profile: dict[str, object],
    lr_summary: pd.DataFrame,
    positive_summary: pd.DataFrame,
    cell_type_summary: pd.DataFrame,
) -> dict[str, object]:
    present_controls = positive_summary[positive_summary["present_in_lr_table"] == True]  # noqa: E712
    top_decile = present_controls[present_controls["rank_percentile_by_prob_mean"] <= 0.10]
    top_pairs = cell_type_summary.head(3)
    top_pair_text = "; ".join(
        f"{row.sender_cell_type}->{row.receiver_cell_type}:{int(row.n_top_edges)}"
        for row in top_pairs.itertuples(index=False)
    )
    best_control = (
        present_controls.sort_values("rank_by_prob_mean").iloc[0].to_dict() if not present_controls.empty else {}
    )
    return {
        "goal_id": "4",
        "dataset": dataset_profile["dataset"],
        "evidence_status": "initial_high_resolution_evidence",
        "analysis_label": dataset_profile["analysis_label"],
        "is_fresh_phase2_default_inference": dataset_profile["is_fresh_phase2_default_inference"],
        "n_cells": dataset_profile["n_cells"],
        "n_cell_types": dataset_profile["n_cell_types"],
        "n_prediction_rows_processed": dataset_profile["n_prediction_rows_processed"],
        "n_lr_pairs_ranked": int(len(lr_summary)),
        "n_top_edges_analyzed": dataset_profile["n_top_edges_output"],
        "n_positive_controls_present": int(len(present_controls)),
        "n_positive_controls_top_decile": int(len(top_decile)),
        "best_positive_control": best_control.get("control_name", ""),
        "best_positive_control_rank": best_control.get("rank_by_prob_mean", ""),
        "median_positive_control_rank_percentile": float(present_controls["rank_percentile_by_prob_mean"].median())
        if not present_controls.empty
        else math.nan,
        "top_cell_type_pairs": top_pair_text,
        "wall_clock_seconds": dataset_profile["wall_clock_seconds"],
        "peak_rss_mb": dataset_profile["peak_rss_mb"],
        "interpretation": (
            "Initial high-resolution real-data evidence from an existing tonsil prediction stream; "
            "requires fresh Phase 2 default inference before manuscript-level claims."
        ),
    }


def _expected_fraction(
    edges: pd.DataFrame,
    expected_senders: set[str],
    expected_receivers: set[str],
    *,
    allow_reverse: bool,
) -> float:
    if edges.empty:
        return 0.0
    forward = edges["sender_cell_type"].isin(expected_senders) & edges["receiver_cell_type"].isin(expected_receivers)
    if allow_reverse:
        reverse = edges["sender_cell_type"].isin(expected_receivers) & edges["receiver_cell_type"].isin(expected_senders)
        return float((forward | reverse).mean())
    return float(forward.mean())


def _specificity(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0 or len(counts) <= 1:
        return 1.0
    probs = counts / total
    entropy = float(-(probs * np.log(probs)).sum())
    return float(1.0 - entropy / math.log(len(counts)))


def _coordinate_dispersion(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0:
        return math.nan
    cx = float(np.mean(x))
    cy = float(np.mean(y))
    return float(np.sqrt(np.mean((x - cx) ** 2 + (y - cy) ** 2)))


def _canonical_pair(a: object, b: object) -> str:
    left, right = sorted([str(a), str(b)])
    return f"{left}-{right}"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    main()
