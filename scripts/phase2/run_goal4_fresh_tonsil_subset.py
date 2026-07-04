#!/usr/bin/env python
"""Run fresh Phase 2 default SpatialSelf inference on a tonsil subset.

This creates a benchmark-style real-data replicate from the local human tonsil
dataset, trains the current Phase 2 default candidate, and reports real-data
validation summaries. Because the dataset has no ground truth, this script does
not compute AUROC/AUPR.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.run_spatialself import (  # noqa: E402
    _build_training_data,
    _indexed_pairwise_ranking_loss,
    _weighted_bce_or_focal_loss,
)
from src.spitialself.model_factory import build_model  # noqa: E402


@dataclass(frozen=True)
class ControlPrior:
    control_name: str
    control_class: str
    ligand: str
    receptor: str
    mechanism: str
    active_niche: str
    spatial_range: float
    expected_senders: frozenset[str]
    expected_receivers: frozenset[str]
    false_positive_high_expression: bool = False


POSITIVE_CONTROLS = (
    ControlPrior(
        "CXCL13-CXCR5",
        "tonsil_positive_prior",
        "CXCL13",
        "CXCR5",
        "long_range_chemokine_niche",
        "germinal_center",
        260.0,
        frozenset({"FDC", "T_follicular_helper", "B_germinal_center"}),
        frozenset({"B_germinal_center", "B_naive", "B_memory", "T_follicular_helper"}),
    ),
    ControlPrior(
        "CXCL12-CXCR4",
        "tonsil_positive_prior",
        "CXCL12",
        "CXCR4",
        "long_range_chemokine_niche",
        "any",
        260.0,
        frozenset({"FDC", "myeloid", "mDC"}),
        frozenset({"B_germinal_center", "B_naive", "B_memory", "T_CD4", "T_follicular_helper"}),
    ),
    ControlPrior(
        "CD40-CD40LG",
        "tonsil_positive_prior",
        "CD40LG",
        "CD40",
        "juxtacrine_contact",
        "germinal_center",
        70.0,
        frozenset({"T_CD4", "T_follicular_helper"}),
        frozenset({"B_germinal_center", "B_naive", "B_memory", "FDC"}),
    ),
    ControlPrior(
        "ICOSLG-ICOS",
        "tonsil_positive_prior",
        "ICOSLG",
        "ICOS",
        "juxtacrine_contact",
        "germinal_center",
        70.0,
        frozenset({"B_germinal_center", "B_naive", "B_memory", "FDC", "mDC"}),
        frozenset({"T_follicular_helper", "T_CD4"}),
    ),
    ControlPrior(
        "TNFSF13B-TNFRSF13C",
        "tonsil_positive_prior",
        "TNFSF13B",
        "TNFRSF13C",
        "short_medium_paracrine",
        "germinal_center",
        150.0,
        frozenset({"myeloid", "mDC", "FDC"}),
        frozenset({"B_germinal_center", "B_naive", "B_memory", "plasma"}),
    ),
    ControlPrior(
        "CCL21-CCR7",
        "tonsil_positive_prior",
        "CCL21",
        "CCR7",
        "long_range_chemokine_niche",
        "T_zone",
        260.0,
        frozenset({"mDC", "FDC", "T_CD4"}),
        frozenset({"T_CD4", "T_CD8", "T_follicular_helper", "mDC"}),
    ),
)

NEGATIVE_CONTROLS = (
    ControlPrior(
        "ACTB-GAPDH",
        "housekeeping_negative_control",
        "ACTB",
        "GAPDH",
        "false_positive_high_expression",
        "any",
        260.0,
        frozenset(),
        frozenset(),
        True,
    ),
    ControlPrior(
        "GAPDH-ITGB1",
        "housekeeping_negative_control",
        "GAPDH",
        "ITGB1",
        "false_positive_high_expression",
        "any",
        260.0,
        frozenset(),
        frozenset(),
        True,
    ),
    ControlPrior(
        "ACTB-EGFR",
        "housekeeping_negative_control",
        "ACTB",
        "EGFR",
        "false_positive_high_expression",
        "any",
        260.0,
        frozenset(),
        frozenset(),
        True,
    ),
)

ALL_CONTROLS = POSITIVE_CONTROLS + NEGATIVE_CONTROLS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spatial", default="high_spatial/data/HumanTonsil_spatial.csv")
    parser.add_argument("--expression", default="high_spatial/data/HumanTonsil_expression.csv.gz")
    parser.add_argument("--lr-table", default="high_spatial/visium_lr_benchmark/lr_table.csv")
    parser.add_argument("--output-dir", default="outputs/phase2/goal4_high_resolution/tonsil_fresh_wavelet_calibrated_subset")
    parser.add_argument("--n-cells", type=int, default=720)
    parser.add_argument("--n-background-lr", type=int, default=36)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--k-nn", type=int, default=12)
    parser.add_argument("--h-dim", type=int, default=64)
    parser.add_argument("--lr-dim", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-train-samples", type=int, default=8000)
    parser.add_argument("--positive-quantile", type=float, default=0.90)
    parser.add_argument("--negative-quantile", type=float, default=0.20)
    parser.add_argument("--operating-quantile", type=float, default=0.94)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    started = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    output_dir = PROJECT_ROOT / args.output_dir
    replicate_dir = output_dir / "constructed_replicate"
    output_dir.mkdir(parents=True, exist_ok=True)
    replicate_dir.mkdir(parents=True, exist_ok=True)

    spatial = _read_spatial(PROJECT_ROOT / args.spatial)
    cells = _select_cells(spatial, n_cells=args.n_cells, rng=rng)
    lr_source = pd.read_csv(PROJECT_ROOT / args.lr_table)
    all_lr_genes = set(lr_source["ligand"].astype(str)) | set(lr_source["receptor"].astype(str))
    all_lr_genes |= {control.ligand for control in ALL_CONTROLS}
    all_lr_genes |= {control.receptor for control in ALL_CONTROLS}
    expression = _read_expression(
        PROJECT_ROOT / args.expression,
        cell_ids=cells["cell_id"].tolist(),
        wanted_genes=all_lr_genes,
    )
    lr_pairs = _select_lr_pairs(
        lr_source,
        expression,
        n_background_lr=args.n_background_lr,
    )
    candidates = _build_candidates(cells, expression, lr_pairs)

    cells.to_csv(replicate_dir / "cells.csv", index=False)
    expression.to_csv(replicate_dir / "expression.csv", index=False)
    lr_pairs.to_csv(replicate_dir / "lr_pairs.csv", index=False)
    candidates.to_csv(replicate_dir / "candidate_interactions.csv", index=False)

    train_started = time.perf_counter()
    prediction = _train_and_score(
        cells,
        expression,
        lr_pairs,
        candidates,
        output_dir=output_dir,
        epochs=args.epochs,
        k_nn=args.k_nn,
        h_dim=args.h_dim,
        lr_dim=args.lr_dim,
        learning_rate=args.learning_rate,
        max_train_samples=args.max_train_samples,
        positive_quantile=args.positive_quantile,
        negative_quantile=args.negative_quantile,
        seed=args.seed,
    )
    train_seconds = time.perf_counter() - train_started
    prediction_path = output_dir / "tonsil_fresh_spatialself_wavelet_predictions.csv"
    prediction.to_csv(prediction_path, index=False)

    annotated = candidates.merge(prediction, on="candidate_id", how="inner", validate="one_to_one")
    lr_summary = _summarize_lr(annotated, cells, operating_quantile=args.operating_quantile)
    cell_type_summary = _summarize_cell_type_pairs(annotated, operating_quantile=args.operating_quantile)
    control_summary = _summarize_controls(lr_summary, annotated, operating_quantile=args.operating_quantile)
    dataset_profile = _dataset_profile(
        args=args,
        cells=cells,
        lr_pairs=lr_pairs,
        candidates=candidates,
        elapsed_seconds=time.perf_counter() - started,
        train_seconds=train_seconds,
    )
    goal_summary = _goal_summary(dataset_profile, control_summary, cell_type_summary)

    lr_summary.to_csv(output_dir / "tonsil_fresh_lr_summary.csv", index=False)
    cell_type_summary.to_csv(output_dir / "tonsil_fresh_cell_type_pair_summary.csv", index=False)
    control_summary.to_csv(output_dir / "tonsil_fresh_control_summary.csv", index=False)
    pd.DataFrame([dataset_profile]).to_csv(output_dir / "tonsil_fresh_dataset_profile.csv", index=False)
    pd.DataFrame([goal_summary]).to_csv(output_dir / "tonsil_fresh_goal4_summary.csv", index=False)
    (output_dir / "tonsil_fresh_manifest.json").write_text(
        json.dumps(
            {
                "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
                "parameters": vars(args),
                "constructed_replicate": str(replicate_dir.relative_to(PROJECT_ROOT)),
                "prediction_path": str(prediction_path.relative_to(PROJECT_ROOT)),
                "dataset_profile": dataset_profile,
                "goal_summary": goal_summary,
                "control_priors": [_control_to_dict(control) for control in ALL_CONTROLS],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Fresh Goal 4 profile:  {output_dir / 'tonsil_fresh_dataset_profile.csv'}")
    print(f"Fresh Goal 4 summary:  {output_dir / 'tonsil_fresh_goal4_summary.csv'}")
    print(f"Fresh LR summary:      {output_dir / 'tonsil_fresh_lr_summary.csv'}")
    print(f"Fresh control summary: {output_dir / 'tonsil_fresh_control_summary.csv'}")


def _read_spatial(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not df.empty and str(df.iloc[0]["NAME"]) == "TYPE":
        df = df.iloc[1:].copy()
    df = df.rename(columns={"NAME": "cell_id", "X": "x", "Y": "y"})
    df["x"] = pd.to_numeric(df["x"])
    df["y"] = pd.to_numeric(df["y"])
    df["dominant_cell_type"] = df["cell_type"].astype(str)
    df["cell_type"] = df["cell_type"].astype(str)
    df["region"] = df["cell_type"].map(_region_from_cell_type)
    df["niche"] = df["cell_type"].map(_niche_from_cell_type)
    return df[["cell_id", "x", "y", "dominant_cell_type", "cell_type", "region", "niche"]].reset_index(drop=True)


def _select_cells(spatial: pd.DataFrame, *, n_cells: int, rng: np.random.Generator) -> pd.DataFrame:
    quotas = {
        "B_germinal_center": 0.26,
        "FDC": 0.14,
        "T_follicular_helper": 0.14,
        "B_naive": 0.12,
        "T_CD4": 0.12,
        "B_memory": 0.08,
        "myeloid": 0.06,
        "mDC": 0.04,
        "plasma": 0.03,
        "T_CD8": 0.01,
    }
    selected = []
    for cell_type, fraction in quotas.items():
        subset = spatial[spatial["cell_type"] == cell_type]
        if subset.empty:
            continue
        take = min(len(subset), max(int(round(n_cells * fraction)), 1))
        selected.append(subset.iloc[rng.choice(len(subset), size=take, replace=False)])
    cells = pd.concat(selected, ignore_index=True).drop_duplicates("cell_id")
    if len(cells) < n_cells:
        remaining = spatial[~spatial["cell_id"].isin(cells["cell_id"])]
        take = min(len(remaining), n_cells - len(cells))
        if take > 0:
            cells = pd.concat(
                [cells, remaining.iloc[rng.choice(len(remaining), size=take, replace=False)]],
                ignore_index=True,
            )
    if len(cells) > n_cells:
        cells = cells.iloc[rng.choice(len(cells), size=n_cells, replace=False)].copy()
    return cells.sort_values("cell_id").reset_index(drop=True)


def _read_expression(path: Path, *, cell_ids: list[str], wanted_genes: set[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    gene_column = header[0]
    missing_cells = sorted(set(cell_ids) - set(header))
    if missing_cells:
        raise ValueError(f"Expression matrix is missing selected cells: {missing_cells[:5]}")
    raw = pd.read_csv(path, usecols=[gene_column, *cell_ids])
    raw = raw[raw[gene_column].astype(str).isin(wanted_genes)].copy()
    if raw.empty:
        raise ValueError("No selected LR/control genes were found in the expression matrix.")
    raw = raw.groupby(gene_column, as_index=True).sum(numeric_only=True)
    expression = raw.T.reset_index().rename(columns={"index": "cell_id"})
    for column in expression.columns:
        if column != "cell_id":
            expression[column] = pd.to_numeric(expression[column], errors="coerce").fillna(0.0)
    return expression


def _select_lr_pairs(lr_source: pd.DataFrame, expression: pd.DataFrame, *, n_background_lr: int) -> pd.DataFrame:
    genes = set(expression.columns) - {"cell_id"}
    expr = expression.drop(columns=["cell_id"])
    means = expr.mean(axis=0)
    detection = (expr > 0).mean(axis=0)
    source = lr_source[
        lr_source["ligand"].astype(str).isin(genes) & lr_source["receptor"].astype(str).isin(genes)
    ].copy()
    source["canonical_pair"] = [_canonical_pair(row.ligand, row.receptor) for row in source.itertuples(index=False)]
    forced_pairs = {_canonical_pair(control.ligand, control.receptor) for control in ALL_CONTROLS}
    source["expression_detectability_score"] = [
        float(np.log1p(means[row.ligand]) * np.log1p(means[row.receptor]) * (0.5 + detection[row.ligand]) * (0.5 + detection[row.receptor]))
        for row in source.itertuples(index=False)
    ]
    selected_rows = []
    seen = set()
    for control in ALL_CONTROLS:
        if control.ligand not in genes or control.receptor not in genes:
            continue
        match = source[source["canonical_pair"] == _canonical_pair(control.ligand, control.receptor)]
        if match.empty:
            selected_rows.append(_lr_pair_from_control(control, source_lr_id=f"synthetic_control_{control.control_name}"))
        else:
            row = match.sort_values("expression_detectability_score", ascending=False).iloc[0].to_dict()
            row.update(_lr_metadata_from_control(control))
            selected_rows.append(row)
        seen.add(_canonical_pair(control.ligand, control.receptor))
    background = source[~source["canonical_pair"].isin(seen)].sort_values("expression_detectability_score", ascending=False)
    for row in background.head(int(n_background_lr)).to_dict(orient="records"):
        row.update(_infer_lr_metadata(row["ligand"], row["receptor"]))
        selected_rows.append(row)
        seen.add(row["canonical_pair"])
    rows = []
    for idx, row in enumerate(selected_rows):
        ligand = str(row["ligand"])
        receptor = str(row["receptor"])
        rows.append(
            {
                "lr_id": f"tonsil_lr_{idx:03d}_{ligand}_{receptor}",
                "source_lr_id": row.get("lr_id", ""),
                "ligand": ligand,
                "receptor": receptor,
                "mechanism": row.get("mechanism", "short_medium_paracrine"),
                "spatial_range": float(row.get("spatial_range", 150.0)),
                "active_niche": row.get("active_niche", "any"),
                "control_name": row.get("control_name", ""),
                "control_class": row.get("control_class", "background_lr"),
                "false_positive_high_expression": bool(row.get("false_positive_high_expression", False)),
                "true_lr": not bool(row.get("false_positive_high_expression", False)),
            }
        )
    return pd.DataFrame(rows)


def _build_candidates(cells: pd.DataFrame, expression: pd.DataFrame, lr_pairs: pd.DataFrame) -> pd.DataFrame:
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("scipy is required for fresh tonsil candidate construction") from exc
    coords = cells[["x", "y"]].to_numpy(dtype=float)
    tree = cKDTree(coords)
    max_k = 48
    dist, neigh = tree.query(coords, k=max_k + 1)
    dist = np.asarray(dist)[:, 1:]
    neigh = np.asarray(neigh)[:, 1:]
    edge_cache = {
        "juxtacrine_contact": _edge_frame_from_neighbors(dist, neigh, k=8, max_distance=70.0),
        "short_medium_paracrine": _edge_frame_from_neighbors(dist, neigh, k=20, max_distance=160.0),
        "long_range_chemokine_niche": _edge_frame_from_neighbors(dist, neigh, k=48, max_distance=280.0),
        "false_positive_high_expression": _edge_frame_from_neighbors(dist, neigh, k=32, max_distance=260.0),
    }
    cell_id = cells["cell_id"].to_numpy()
    cell_type = cells["cell_type"].to_numpy()
    niche = cells["niche"].to_numpy()
    expression_index = expression.set_index("cell_id")
    parts = []
    counter = 0
    for lr in lr_pairs.itertuples(index=False):
        edges = edge_cache.get(lr.mechanism, edge_cache["short_medium_paracrine"])
        src = edges["src_idx"].to_numpy(dtype=int)
        dst = edges["dst_idx"].to_numpy(dtype=int)
        ligand_expr = expression_index.loc[cell_id[src], lr.ligand].to_numpy(dtype=float)
        receptor_expr = expression_index.loc[cell_id[dst], lr.receptor].to_numpy(dtype=float)
        type_rule = _type_rule(lr, cell_type[src], cell_type[dst])
        range_rule = edges["distance"].to_numpy(dtype=float) <= float(lr.spatial_range)
        niche_rule = _niche_rule(str(lr.active_niche), niche[src], niche[dst])
        coexp = np.log1p(ligand_expr) * np.log1p(receptor_expr)
        bandwidth = max(float(np.median(edges["distance"])), 1e-6)
        score = coexp * np.exp(-edges["distance"].to_numpy(dtype=float) / bandwidth)
        n = len(edges)
        part = pd.DataFrame(
            {
                "candidate_id": [f"tonsil_fresh_cand_{counter + i:08d}" for i in range(n)],
                "sender_id": cell_id[src],
                "receiver_id": cell_id[dst],
                "lr_id": lr.lr_id,
                "ligand": lr.ligand,
                "receptor": lr.receptor,
                "mechanism": lr.mechanism,
                "distance": edges["distance"].to_numpy(dtype=float),
                "sender_cell_type": cell_type[src],
                "receiver_cell_type": cell_type[dst],
                "sender_niche": niche[src],
                "receiver_niche": niche[dst],
                "weak_signal": False,
                "false_positive_high_expression": bool(lr.false_positive_high_expression),
                "type_rule_satisfied": type_rule,
                "range_rule_satisfied": range_rule,
                "niche_rule_satisfied": niche_rule,
                "ligand_expr": ligand_expr,
                "receptor_expr": receptor_expr,
                "coexpression_distance_score": score,
            }
        )
        counter += n
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _train_and_score(
    cells: pd.DataFrame,
    expression: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    output_dir: Path,
    epochs: int,
    k_nn: int,
    h_dim: int,
    lr_dim: int,
    learning_rate: float,
    max_train_samples: int,
    positive_quantile: float,
    negative_quantile: float,
    seed: int,
) -> pd.DataFrame:
    torch.manual_seed(seed)
    np.random.seed(seed)
    data, tensors = _build_training_data(
        cells,
        expression,
        lr_pairs,
        candidates,
        k_nn=k_nn,
        positive_quantile=positive_quantile,
        negative_quantile=negative_quantile,
        pseudo_label_strategy="lr_prior_lr_capped_mechanism_niche_expression_distance",
        soft_labels=False,
        weak_signal_reweighting=1.0,
        weak_positive_fraction=0.0,
        weak_positive_label=1.0,
        weak_positive_mode="all",
        train_sampling_strategy="balanced_global",
        lr_cap_positive=50,
        lr_cap_negative=200,
        use_spatial_graph=True,
        use_expression_features=True,
        use_cell_type_features=True,
        use_niche_features=True,
        use_coordinate_features=True,
        use_edge_prior_features=True,
        max_train_samples=max_train_samples,
        seed=seed,
    )
    model = build_model(
        "spatialself_wavelet",
        in_dim=data.x.shape[1],
        h_dim=h_dim,
        lr_dim=lr_dim,
        n_lr=len(lr_pairs),
        distance_scale=max(float(candidates["distance"].quantile(0.95)), 1e-6),
        use_lr_embedding=True,
        lr_encoding_mode="learned",
        use_distance=True,
        edge_feature_dim=int(tensors["edge_features"].shape[1]),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    train_idx = tensors["train_idx"]
    labels = tensors["pseudo_label"][train_idx]
    sample_weight = tensors["sample_weight"][train_idx]
    for _epoch in range(int(epochs)):
        model.train()
        optimizer.zero_grad()
        h = model.encode_nodes(data)
        logits = model.score_edges(
            h,
            tensors["src_idx"][train_idx],
            tensors["dst_idx"][train_idx],
            tensors["lr_idx"][train_idx],
            tensors["distance"][train_idx].unsqueeze(-1),
            tensors["edge_features"][train_idx],
        )
        loss = _weighted_bce_or_focal_loss(
            logits,
            labels,
            weight=sample_weight,
            focal_gamma=0.0,
            focal_alpha=0.5,
        )
        ranking_pos = tensors["ranking_pos_idx"]
        ranking_neg = tensors["ranking_neg_idx"]
        if ranking_pos.numel() and ranking_neg.numel():
            pos_logits = model.score_edges(
                h,
                tensors["src_idx"][ranking_pos],
                tensors["dst_idx"][ranking_pos],
                tensors["lr_idx"][ranking_pos],
                tensors["distance"][ranking_pos].unsqueeze(-1),
                tensors["edge_features"][ranking_pos],
            )
            neg_logits = model.score_edges(
                h,
                tensors["src_idx"][ranking_neg],
                tensors["dst_idx"][ranking_neg],
                tensors["lr_idx"][ranking_neg],
                tensors["distance"][ranking_neg].unsqueeze(-1),
                tensors["edge_features"][ranking_neg],
            )
            loss = loss + 0.35 * _indexed_pairwise_ranking_loss(pos_logits, neg_logits, margin=0.5)
        loss.backward()
        optimizer.step()

    scores = []
    batch_size = 65536
    model.eval()
    with torch.no_grad():
        h = model.encode_nodes(data)
        for start in range(0, len(candidates), batch_size):
            end = min(start + batch_size, len(candidates))
            logits = model.score_edges(
                h,
                tensors["src_idx"][start:end],
                tensors["dst_idx"][start:end],
                tensors["lr_idx"][start:end],
                tensors["distance"][start:end].unsqueeze(-1),
                tensors["edge_features"][start:end],
            )
            scores.append(torch.sigmoid(logits).cpu().numpy())
    model_path = output_dir / "tonsil_fresh_spatialself_wavelet_model.pt"
    torch.save(model.state_dict(), model_path)
    return pd.DataFrame(
        {
            "candidate_id": candidates["candidate_id"].to_numpy(),
            "method": "wavelet_calibrated_fdr_fresh_tonsil_subset",
            "score": np.concatenate(scores),
        }
    )


def _summarize_lr(annotated: pd.DataFrame, cells: pd.DataFrame, *, operating_quantile: float) -> pd.DataFrame:
    threshold = float(annotated["score"].quantile(operating_quantile))
    top = annotated[annotated["score"] >= threshold].copy()
    top["cell_type_pair"] = top["sender_cell_type"].astype(str) + "->" + top["receiver_cell_type"].astype(str)
    all_dispersion = _coordinate_dispersion(cells["x"].to_numpy(), cells["y"].to_numpy())
    rows = []
    for lr_id, group in annotated.groupby("lr_id", sort=False):
        top_group = top[top["lr_id"] == lr_id]
        pair_counts = top_group["cell_type_pair"].value_counts()
        receiver_dispersion = _receiver_dispersion(top_group, cells)
        rows.append(
            {
                "lr_id": lr_id,
                "ligand": group["ligand"].iloc[0],
                "receptor": group["receptor"].iloc[0],
                "mechanism": group["mechanism"].iloc[0],
                "control_class": _control_class(group["ligand"].iloc[0], group["receptor"].iloc[0]),
                "n_candidate_edges": int(len(group)),
                "score_mean": float(group["score"].mean()),
                "score_max": float(group["score"].max()),
                "score_p95": float(group["score"].quantile(0.95)),
                "distance_mean": float(group["distance"].mean()),
                "top_edge_count": int(len(top_group)),
                "top_score_mean": float(top_group["score"].mean()) if len(top_group) else 0.0,
                "top_distance_median": float(top_group["distance"].median()) if len(top_group) else 0.0,
                "dominant_cell_type_pair": str(pair_counts.index[0]) if len(pair_counts) else "",
                "dominant_cell_type_pair_fraction": float(pair_counts.iloc[0] / len(top_group)) if len(pair_counts) else 0.0,
                "cell_type_specificity": _specificity(pair_counts.to_numpy(dtype=float)),
                "receiver_spatial_concentration": float(1.0 - receiver_dispersion / all_dispersion)
                if all_dispersion > 0 and math.isfinite(receiver_dispersion)
                else 0.0,
            }
        )
    summary = pd.DataFrame(rows)
    summary["canonical_pair"] = [_canonical_pair(row.ligand, row.receptor) for row in summary.itertuples(index=False)]
    summary["rank_by_score_mean"] = summary["score_mean"].rank(method="min", ascending=False).astype(int)
    summary["rank_by_score_max"] = summary["score_max"].rank(method="min", ascending=False).astype(int)
    summary["rank_by_top_edge_count"] = summary["top_edge_count"].rank(method="min", ascending=False).astype(int)
    return summary.sort_values(["rank_by_score_mean", "lr_id"]).reset_index(drop=True)


def _summarize_cell_type_pairs(annotated: pd.DataFrame, *, operating_quantile: float) -> pd.DataFrame:
    threshold = float(annotated["score"].quantile(operating_quantile))
    top = annotated[annotated["score"] >= threshold]
    if top.empty:
        return pd.DataFrame()
    summary = (
        top.groupby(["sender_cell_type", "receiver_cell_type"], observed=True)
        .agg(
            n_top_edges=("score", "size"),
            mean_score=("score", "mean"),
            max_score=("score", "max"),
            median_distance=("distance", "median"),
            n_lr_pairs=("lr_id", "nunique"),
        )
        .reset_index()
    )
    summary["top_edge_fraction"] = summary["n_top_edges"] / summary["n_top_edges"].sum()
    return summary.sort_values(["n_top_edges", "mean_score"], ascending=[False, False]).reset_index(drop=True)


def _summarize_controls(lr_summary: pd.DataFrame, annotated: pd.DataFrame, *, operating_quantile: float) -> pd.DataFrame:
    threshold = float(annotated["score"].quantile(operating_quantile))
    top = annotated[annotated["score"] >= threshold].copy()
    top["canonical_pair"] = [_canonical_pair(row.ligand, row.receptor) for row in top.itertuples(index=False)]
    n_lr = len(lr_summary)
    rows = []
    for control in ALL_CONTROLS:
        canonical = _canonical_pair(control.ligand, control.receptor)
        matches = lr_summary[lr_summary["canonical_pair"] == canonical]
        present = not matches.empty
        best = matches.sort_values("rank_by_score_mean").iloc[0].to_dict() if present else {}
        top_match = top[top["canonical_pair"] == canonical]
        rows.append(
            {
                "control_name": control.control_name,
                "control_class": control.control_class,
                "canonical_pair": canonical,
                "present_in_fresh_subset": bool(present),
                "lr_id": best.get("lr_id", ""),
                "rank_by_score_mean": int(best.get("rank_by_score_mean", 0)) if present else 0,
                "rank_percentile_by_score_mean": float(best.get("rank_by_score_mean", math.nan)) / n_lr
                if present and n_lr
                else math.nan,
                "rank_by_top_edge_count": int(best.get("rank_by_top_edge_count", 0)) if present else 0,
                "score_mean": float(best.get("score_mean", math.nan)) if present else math.nan,
                "score_max": float(best.get("score_max", math.nan)) if present else math.nan,
                "top_edge_count": int(best.get("top_edge_count", 0)) if present else 0,
                "dominant_cell_type_pair": best.get("dominant_cell_type_pair", ""),
                "cell_type_specificity": float(best.get("cell_type_specificity", math.nan)) if present else math.nan,
                "receiver_spatial_concentration": float(best.get("receiver_spatial_concentration", math.nan))
                if present
                else math.nan,
                "directional_expected_cell_type_fraction": _expected_fraction(
                    top_match,
                    control.expected_senders,
                    control.expected_receivers,
                    allow_reverse=False,
                ),
                "unordered_expected_cell_type_fraction": _expected_fraction(
                    top_match,
                    control.expected_senders,
                    control.expected_receivers,
                    allow_reverse=True,
                ),
            }
        )
    return pd.DataFrame(rows)


def _dataset_profile(
    *,
    args: argparse.Namespace,
    cells: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    candidates: pd.DataFrame,
    elapsed_seconds: float,
    train_seconds: float,
) -> dict[str, object]:
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    if platform.system() == "Darwin":
        peak_mb /= 1024.0
    return {
        "dataset": "human_tonsil_slide_tags",
        "analysis_label": "fresh_wavelet_calibrated_fdr_subset",
        "is_fresh_phase2_default_inference": True,
        "model_name": "spatialself_wavelet",
        "phase2_default_candidate": "wavelet_calibrated_fdr",
        "n_cells": int(len(cells)),
        "n_cell_types": int(cells["cell_type"].nunique()),
        "n_lr_pairs": int(len(lr_pairs)),
        "n_candidate_edges": int(len(candidates)),
        "n_positive_controls": int(sum(row.control_class == "tonsil_positive_prior" for row in ALL_CONTROLS)),
        "n_negative_controls": int(sum(row.control_class != "tonsil_positive_prior" for row in ALL_CONTROLS)),
        "epochs": int(args.epochs),
        "k_nn": int(args.k_nn),
        "max_train_samples": int(args.max_train_samples),
        "operating_quantile": float(args.operating_quantile),
        "wall_clock_seconds": float(elapsed_seconds),
        "train_and_inference_seconds": float(train_seconds),
        "peak_rss_mb": float(peak_mb),
    }


def _goal_summary(
    dataset_profile: dict[str, object],
    control_summary: pd.DataFrame,
    cell_type_summary: pd.DataFrame,
) -> dict[str, object]:
    positives = control_summary[control_summary["control_class"] == "tonsil_positive_prior"]
    negatives = control_summary[control_summary["control_class"] != "tonsil_positive_prior"]
    pos_top_decile = positives[positives["rank_percentile_by_score_mean"] <= 0.10]
    neg_top_decile = negatives[negatives["rank_percentile_by_score_mean"] <= 0.10]
    best_pos = positives.sort_values("rank_by_score_mean").iloc[0].to_dict() if not positives.empty else {}
    best_neg = negatives.sort_values("rank_by_score_mean").iloc[0].to_dict() if not negatives.empty else {}
    top_pairs = "; ".join(
        f"{row.sender_cell_type}->{row.receiver_cell_type}:{int(row.n_top_edges)}"
        for row in cell_type_summary.head(3).itertuples(index=False)
    )
    return {
        "goal_id": "4",
        "dataset": dataset_profile["dataset"],
        "evidence_status": "fresh_subset_initial",
        "analysis_label": dataset_profile["analysis_label"],
        "is_fresh_phase2_default_inference": True,
        "n_cells": dataset_profile["n_cells"],
        "n_lr_pairs": dataset_profile["n_lr_pairs"],
        "n_candidate_edges": dataset_profile["n_candidate_edges"],
        "n_positive_controls_top_decile": int(len(pos_top_decile)),
        "n_negative_controls_top_decile": int(len(neg_top_decile)),
        "best_positive_control": best_pos.get("control_name", ""),
        "best_positive_control_rank": best_pos.get("rank_by_score_mean", ""),
        "best_negative_control": best_neg.get("control_name", ""),
        "best_negative_control_rank": best_neg.get("rank_by_score_mean", ""),
        "median_positive_rank_percentile": float(positives["rank_percentile_by_score_mean"].median())
        if not positives.empty
        else math.nan,
        "median_negative_rank_percentile": float(negatives["rank_percentile_by_score_mean"].median())
        if not negatives.empty
        else math.nan,
        "top_cell_type_pairs": top_pairs,
        "wall_clock_seconds": dataset_profile["wall_clock_seconds"],
        "peak_rss_mb": dataset_profile["peak_rss_mb"],
        "interpretation": (
            "Fresh high-resolution subset run for wiring and initial biological controls; "
            "requires larger subset/full data and downstream validation before manuscript claims."
        ),
    }


def _edge_frame_from_neighbors(dist: np.ndarray, neigh: np.ndarray, *, k: int, max_distance: float) -> pd.DataFrame:
    keep_dist = dist[:, :k]
    keep_neigh = neigh[:, :k]
    src = np.repeat(np.arange(dist.shape[0]), k)
    dst = keep_neigh.reshape(-1)
    flat_dist = keep_dist.reshape(-1)
    mask = np.isfinite(flat_dist) & (flat_dist > 0) & (flat_dist <= float(max_distance))
    return pd.DataFrame({"src_idx": src[mask].astype(np.int64), "dst_idx": dst[mask].astype(np.int64), "distance": flat_dist[mask]})


def _type_rule(lr: object, sender_types: np.ndarray, receiver_types: np.ndarray) -> np.ndarray:
    control = _find_control(str(lr.ligand), str(lr.receptor))
    if bool(lr.false_positive_high_expression):
        return np.zeros(len(sender_types), dtype=bool)
    if control is None or not control.expected_senders or not control.expected_receivers:
        return np.ones(len(sender_types), dtype=bool)
    forward = np.isin(sender_types, list(control.expected_senders)) & np.isin(receiver_types, list(control.expected_receivers))
    reverse = np.isin(sender_types, list(control.expected_receivers)) & np.isin(receiver_types, list(control.expected_senders))
    return forward | reverse


def _niche_rule(active_niche: str, sender_niche: np.ndarray, receiver_niche: np.ndarray) -> np.ndarray:
    if active_niche in {"", "any", "nan", "None"}:
        return np.ones(len(sender_niche), dtype=bool)
    return (sender_niche == active_niche) & (receiver_niche == active_niche)


def _expected_fraction(
    edges: pd.DataFrame,
    expected_senders: frozenset[str],
    expected_receivers: frozenset[str],
    *,
    allow_reverse: bool,
) -> float:
    if edges.empty or not expected_senders or not expected_receivers:
        return 0.0
    forward = edges["sender_cell_type"].isin(expected_senders) & edges["receiver_cell_type"].isin(expected_receivers)
    if allow_reverse:
        reverse = edges["sender_cell_type"].isin(expected_receivers) & edges["receiver_cell_type"].isin(expected_senders)
        return float((forward | reverse).mean())
    return float(forward.mean())


def _receiver_dispersion(top_group: pd.DataFrame, cells: pd.DataFrame) -> float:
    if top_group.empty:
        return math.nan
    coords = cells.set_index("cell_id").loc[top_group["receiver_id"], ["x", "y"]].to_numpy(dtype=float)
    return _coordinate_dispersion(coords[:, 0], coords[:, 1])


def _coordinate_dispersion(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0:
        return math.nan
    cx = float(np.mean(x))
    cy = float(np.mean(y))
    return float(np.sqrt(np.mean((x - cx) ** 2 + (y - cy) ** 2)))


def _specificity(counts: np.ndarray) -> float:
    if len(counts) <= 1:
        return 1.0
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    probs = counts / total
    entropy = float(-(probs * np.log(probs)).sum())
    return float(1.0 - entropy / math.log(len(counts)))


def _infer_lr_metadata(ligand: str, receptor: str) -> dict[str, object]:
    ligand = str(ligand)
    receptor = str(receptor)
    chemokine_prefixes = ("CCL", "CXCL", "XCL", "CX3CL")
    contact_markers = ("NOTCH", "DLL", "JAG", "CDH", "ICAM", "VCAM", "ITG", "SELE", "SELL", "SELPLG")
    if ligand.startswith(chemokine_prefixes) or receptor.startswith(("CCR", "CXCR", "XCR", "CX3CR")):
        return {"mechanism": "long_range_chemokine_niche", "spatial_range": 260.0, "active_niche": "any"}
    if any(token in ligand or token in receptor for token in contact_markers):
        return {"mechanism": "juxtacrine_contact", "spatial_range": 70.0, "active_niche": "any"}
    return {"mechanism": "short_medium_paracrine", "spatial_range": 150.0, "active_niche": "any"}


def _lr_metadata_from_control(control: ControlPrior) -> dict[str, object]:
    return {
        "mechanism": control.mechanism,
        "spatial_range": control.spatial_range,
        "active_niche": control.active_niche,
        "control_name": control.control_name,
        "control_class": control.control_class,
        "false_positive_high_expression": control.false_positive_high_expression,
    }


def _lr_pair_from_control(control: ControlPrior, *, source_lr_id: str) -> dict[str, object]:
    row = {
        "lr_id": source_lr_id,
        "ligand": control.ligand,
        "receptor": control.receptor,
        "canonical_pair": _canonical_pair(control.ligand, control.receptor),
    }
    row.update(_lr_metadata_from_control(control))
    return row


def _control_to_dict(control: ControlPrior) -> dict[str, object]:
    return {
        "control_name": control.control_name,
        "control_class": control.control_class,
        "ligand": control.ligand,
        "receptor": control.receptor,
        "mechanism": control.mechanism,
        "active_niche": control.active_niche,
        "spatial_range": control.spatial_range,
        "expected_senders": sorted(control.expected_senders),
        "expected_receivers": sorted(control.expected_receivers),
        "false_positive_high_expression": control.false_positive_high_expression,
    }


def _find_control(ligand: str, receptor: str) -> ControlPrior | None:
    canonical = _canonical_pair(ligand, receptor)
    for control in ALL_CONTROLS:
        if _canonical_pair(control.ligand, control.receptor) == canonical:
            return control
    return None


def _control_class(ligand: str, receptor: str) -> str:
    control = _find_control(ligand, receptor)
    return control.control_class if control else "background_lr"


def _canonical_pair(a: object, b: object) -> str:
    left, right = sorted([str(a), str(b)])
    return f"{left}-{right}"


def _region_from_cell_type(cell_type: str) -> str:
    if cell_type in {"B_germinal_center", "FDC", "T_follicular_helper"}:
        return "germinal_center"
    if cell_type.startswith("B_") or cell_type == "plasma":
        return "B_zone"
    if cell_type.startswith("T_"):
        return "T_zone"
    if cell_type in {"myeloid", "mDC", "pDC"}:
        return "myeloid_zone"
    return "other"


def _niche_from_cell_type(cell_type: str) -> str:
    if cell_type in {"B_germinal_center", "FDC", "T_follicular_helper"}:
        return "germinal_center"
    if cell_type in {"T_CD4", "T_CD8", "T_double_neg"}:
        return "T_zone"
    if cell_type in {"B_naive", "B_memory", "plasma"}:
        return "B_zone"
    if cell_type in {"myeloid", "mDC", "pDC"}:
        return "myeloid_zone"
    return "other"


if __name__ == "__main__":
    main()
