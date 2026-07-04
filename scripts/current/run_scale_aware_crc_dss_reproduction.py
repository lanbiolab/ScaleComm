#!/usr/bin/env python
"""Re-run CRC and DSS stories with the scale-aware SpatialSelf candidate.

This script intentionally writes to a fresh output directory and does not
overwrite any frozen baseline or manuscript figure source tables.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark import run_spatialself as rs
from scripts.current import run_mouse_colon_ibd_merfish_stagewise as dss
from scripts.current import run_scale_gate_forcing_experiment as sg


DEFAULT_CRC_DIR = (
    PROJECT_ROOT
    / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/"
    "standardized_inputs/visium_hd_crc_p2crc_rich_targets_subset"
)
DEFAULT_DSS_ROOT = PROJECT_ROOT / "outputs/current_mouse_colon_ibd_merfish/stagewise_replicates"
DEFAULT_OUT = PROJECT_ROOT / "outputs/current_scale_aware_reproduction"

FOCUS_CRC = ["TGFB1-TGFBR2", "SPP1-CD44", "CXCL12-CXCR4", "VEGFA-KDR", "CD274-PDCD1"]
FOCUS_DSS = ["Cxcl10-Cxcr3", "Cxcl9-Cxcr3", "Cxcl12-Cxcr4", "Spp1-Cd44", "Tgfb1-Tgfbr2", "Il1b-Il1r1"]


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def mechanism_to_range_class(mechanism: str, spatial_range: float | None = None) -> str:
    mechanism = str(mechanism)
    if mechanism == "juxtacrine_contact":
        return "contact"
    if mechanism == "short_range_paracrine":
        return "short"
    if mechanism in {"short_medium_paracrine", "medium_range_paracrine"}:
        return "medium"
    if mechanism == "long_range_chemokine_niche":
        return "long_niche"
    if spatial_range is not None and np.isfinite(spatial_range):
        if spatial_range <= 80:
            return "contact"
        if spatial_range <= 160:
            return "short"
        if spatial_range <= 320:
            return "medium"
        return "long_niche"
    return "background"


def train_scale_aware_dataset(dataset_dir: Path, output_dir: Path, args: argparse.Namespace, *, seed: int) -> dict:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = output_dir / "source_data"
    source_dir.mkdir(exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    cells = pd.read_csv(dataset_dir / "cells.csv")
    expression = pd.read_csv(dataset_dir / "expression.csv")
    lr_pairs = pd.read_csv(dataset_dir / "lr_pairs.csv")
    candidates = pd.read_csv(dataset_dir / "candidate_interactions.csv")
    candidates = candidates.copy()
    for col, default in [
        ("weak_signal", False),
        ("false_positive_high_expression", False),
        ("true_lr", True),
        ("type_rule_satisfied", True),
        ("range_rule_satisfied", True),
        ("niche_rule_satisfied", True),
    ]:
        if col not in candidates.columns:
            candidates[col] = default
    if "coexpression_distance_score" not in candidates.columns:
        dist = candidates["distance"].to_numpy(float)
        bandwidth = max(float(np.quantile(dist, 0.5)), 1e-6)
        candidates["coexpression_distance_score"] = (
            np.log1p(candidates["ligand_expr"].to_numpy(float))
            * np.log1p(candidates["receptor_expr"].to_numpy(float))
            * np.exp(-dist / bandwidth)
        )

    lr_pairs = lr_pairs.copy()
    for col, default in [("true_lr", True), ("weak_signal", False), ("false_positive_high_expression", False)]:
        if col not in lr_pairs.columns:
            lr_pairs[col] = default
    if "mechanism" not in lr_pairs.columns:
        lr_pairs["mechanism"] = "short_medium_paracrine"

    lr_design = lr_pairs[
        [col for col in ["lr_id", "mechanism", "true_lr", "weak_signal", "false_positive_high_expression", "spatial_range"] if col in lr_pairs.columns]
    ].copy()
    lr_design["range_class"] = [
        mechanism_to_range_class(row.get("mechanism", ""), row.get("spatial_range", np.nan))
        for row in lr_design.to_dict("records")
    ]

    data, tensors = rs._build_training_data(
        cells,
        expression,
        lr_pairs,
        candidates,
        k_nn=int(args.k_nn),
        positive_quantile=float(args.positive_quantile),
        negative_quantile=float(args.negative_quantile),
        pseudo_label_strategy="mechanism_niche_expression_distance",
        soft_labels=False,
        weak_positive_loss_weight=1.0,
        weak_positive_expression_quantile=0.7,
        train_sampling_strategy="balanced_per_lr",
        lr_cap_positive=int(args.lr_cap_positive),
        lr_cap_negative=int(args.lr_cap_negative),
        use_spatial_graph=True,
        use_expression_features=True,
        use_cell_type_features=True,
        use_niche_features=True,
        use_coordinate_features=True,
        use_edge_prior_features=True,
        max_train_samples=int(args.max_train_samples),
        context_contrastive_negative_mode="random_receiver",
        spatial_null_negative_mode="random_receiver",
        background_guard_targets=None,
        seed=seed,
    )

    model = sg.ScaleMixtureWaveletModel(
        in_dim=data.x.shape[1],
        h_dim=int(args.h_dim),
        lr_dim=int(args.lr_dim),
        n_lr=len(lr_pairs),
        diffusion_steps=(1, 2, 4, 8),
        dropout=float(args.dropout),
        distance_scale=max(float(candidates["distance"].quantile(0.95)), 1e-6),
        edge_feature_dim=int(tensors["edge_features"].shape[1]),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=1e-4)
    train_idx = tensors["train_idx"]
    labels = tensors["pseudo_label"][train_idx]
    sample_weight = tensors["sample_weight"][train_idx]
    aux_target = sg._scale_target(args.variant, lr_pairs, lr_design, candidates, n_scales=model.node_encoder.n_scales)

    for epoch in range(int(args.epochs)):
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
        loss = rs._weighted_bce_or_focal_loss(logits, labels, weight=sample_weight, focal_gamma=0.0, focal_alpha=0.5)
        loss = loss + sg._guard_losses(model, h, tensors)

        lr_ids = torch.arange(len(lr_pairs), dtype=torch.long, device=h.device)
        pred_scale = model.lr_scale_distribution(lr_ids, device=h.device)
        target = aux_target.to(device=h.device, dtype=pred_scale.dtype)
        loss = loss + float(args.scale_target_weight) * sg._kl(pred_scale, target)
        loss = loss + float(args.diversity_weight) * sg._diversity_loss(pred_scale)
        loss = loss + float(args.responsibility_weight) * sg._scale_responsibility_loss(
            model,
            h,
            tensors,
            train_idx,
            labels,
            target,
            margin=float(args.responsibility_margin),
            max_edges=int(args.responsibility_edges),
        )
        loss.backward()
        optimizer.step()
        if args.verbose and (epoch + 1) % int(args.log_every) == 0:
            print(f"{dataset_dir.name}: epoch {epoch + 1}/{args.epochs} loss={float(loss.detach().cpu()):.4f}")

    model.eval()
    with torch.no_grad():
        h = model.encode_nodes(data)
        score_parts = []
        for start in range(0, len(candidates), int(args.score_batch_size)):
            end = min(start + int(args.score_batch_size), len(candidates))
            logits = model.score_edges(
                h,
                tensors["src_idx"][start:end],
                tensors["dst_idx"][start:end],
                tensors["lr_idx"][start:end],
                tensors["distance"][start:end].unsqueeze(-1),
                tensors["edge_features"][start:end],
            )
            score_parts.append(torch.sigmoid(logits).cpu().numpy())
        lr_ids = torch.arange(len(lr_pairs), dtype=torch.long)
        scale = model.lr_scale_distribution(lr_ids).detach().cpu().numpy()

    prediction = pd.DataFrame(
        {
            "candidate_id": candidates["candidate_id"],
            "method": f"spatialself_scale_{args.variant}",
            "score": np.concatenate(score_parts),
        }
    )
    prediction_path = output_dir / f"{dataset_dir.name}__spatialself_scale_{args.variant}_predictions.csv"
    prediction.to_csv(prediction_path, index=False)

    lr_scale = build_lr_scale_table(lr_pairs, lr_design, scale, aux_target.numpy(), str(args.variant))
    lr_scale_path = source_dir / "lr_scale_distribution.csv"
    lr_scale.to_csv(lr_scale_path, index=False)

    rank_table = build_lr_rank_table(candidates, lr_pairs, prediction)
    rank_path = source_dir / "lr_rank_summary.csv"
    rank_table.to_csv(rank_path, index=False)

    summary = {
        "dataset": dataset_dir.name,
        "prediction_path": str(prediction_path.relative_to(PROJECT_ROOT)),
        "lr_rank_summary": str(rank_path.relative_to(PROJECT_ROOT)),
        "lr_scale_distribution": str(lr_scale_path.relative_to(PROJECT_ROOT)),
        "n_cells": int(len(cells)),
        "n_lr_pairs": int(len(lr_pairs)),
        "n_candidates": int(len(candidates)),
        "epochs": int(args.epochs),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_lr_scale_table(
    lr_pairs: pd.DataFrame,
    lr_design: pd.DataFrame,
    scale: np.ndarray,
    target: np.ndarray,
    variant: str,
) -> pd.DataFrame:
    design = lr_design.set_index("lr_id", drop=False)
    idx = np.arange(scale.shape[1], dtype=float)
    rows = []
    for i, row in lr_pairs.reset_index(drop=True).iterrows():
        lr_id = str(row["lr_id"])
        drow = design.loc[lr_id] if lr_id in design.index else {}
        out = {
            "variant": variant,
            "lr_id": lr_id,
            "ligand": row.get("ligand", ""),
            "receptor": row.get("receptor", ""),
            "lr_pair": f"{row.get('ligand', '')}-{row.get('receptor', '')}",
            "mechanism": row.get("mechanism", ""),
            "range_class": drow.get("range_class", "unknown") if hasattr(drow, "get") else "unknown",
            "true_lr": bool(row.get("true_lr", False)),
            "weak_signal": bool(row.get("weak_signal", False)),
            "false_positive_high_expression": bool(row.get("false_positive_high_expression", False)),
            "expected_scale": float((scale[i] * idx).sum()),
            "target_expected_scale": float((target[i] * idx).sum()),
            "local_weight": float(scale[i, :2].sum()),
            "broad_weight": float(scale[i, -2:].sum()),
        }
        for j in range(scale.shape[1]):
            out[f"scale_{j}"] = float(scale[i, j])
            out[f"target_scale_{j}"] = float(target[i, j])
        rows.append(out)
    return pd.DataFrame(rows)


def build_lr_rank_table(candidates: pd.DataFrame, lr_pairs: pd.DataFrame, prediction: pd.DataFrame) -> pd.DataFrame:
    table = candidates.merge(prediction[["candidate_id", "score"]], on="candidate_id", how="left")
    table["score"] = pd.to_numeric(table["score"], errors="coerce").fillna(0.0)
    table["expr_product"] = np.log1p(table["ligand_expr"].astype(float)) * np.log1p(table["receptor_expr"].astype(float))
    group_cols = ["lr_id", "ligand", "receptor", "mechanism"]
    agg = (
        table.groupby(group_cols, dropna=False)
        .agg(
            mean_score=("score", "mean"),
            median_score=("score", "median"),
            q90_score=("score", lambda x: float(np.quantile(x, 0.90))),
            q95_score=("score", lambda x: float(np.quantile(x, 0.95))),
            max_score=("score", "max"),
            n_edges=("score", "size"),
            n_positive_edges=("score", lambda x: int(np.sum(np.asarray(x) > 0))),
            top100_edges=("score", lambda x: int(np.sum(np.asarray(x) >= np.quantile(table["score"], 0.99))) if len(table) else 0),
            expr_product_mean=("expr_product", "mean"),
            expressed_edge_fraction=("expr_product", lambda x: float((np.asarray(x) > 0).mean())),
            false_positive_high_expression=("false_positive_high_expression", "max"),
            true_lr=("true_lr", "max"),
        )
        .reset_index()
        .sort_values("mean_score", ascending=False)
        .reset_index(drop=True)
    )
    agg["rank_by_mean_score"] = np.arange(1, len(agg) + 1)
    agg = agg.sort_values("q95_score", ascending=False).reset_index(drop=True)
    agg["rank_by_q95_score"] = np.arange(1, len(agg) + 1)
    agg = agg.sort_values("mean_score", ascending=False).reset_index(drop=True)
    agg["lr_pair"] = agg["ligand"].astype(str) + "-" + agg["receptor"].astype(str)

    meta_cols = [
        col
        for col in ["lr_id", "spatial_range", "active_niche", "control_name", "control_class", "biological_category", "selected_reason"]
        if col in lr_pairs.columns
    ]
    if meta_cols:
        agg = agg.merge(lr_pairs[meta_cols].drop_duplicates("lr_id"), on="lr_id", how="left")
    return agg


def run_crc(args: argparse.Namespace) -> dict:
    crc_out = resolve(args.output_dir) / "CRC_scale_candidate"
    summary = train_scale_aware_dataset(resolve(args.crc_dataset_dir), crc_out / "model", args, seed=int(args.seed))

    rank = pd.read_csv(crc_out / "model/source_data/lr_rank_summary.csv")
    scale = pd.read_csv(crc_out / "model/source_data/lr_scale_distribution.csv")
    focus = rank[rank["lr_pair"].isin(FOCUS_CRC)].merge(
        scale[["lr_pair", "expected_scale", "local_weight", "broad_weight"]],
        on="lr_pair",
        how="left",
    )
    focus.to_csv(crc_out / "crc_focus_lr_summary.csv", index=False)
    summary["focus_lr_summary"] = str((crc_out / "crc_focus_lr_summary.csv").relative_to(PROJECT_ROOT))
    (crc_out / "crc_reproduction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_dss(args: argparse.Namespace) -> dict:
    root = resolve(args.dss_replicate_root)
    dss_out = resolve(args.output_dir) / "DSS_scale_candidate"
    prediction_root = dss_out / "predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)
    status_rows = []
    for replicate_dir in dss.find_replicates(root):
        stage = dss.stage_from_replicate(replicate_dir)
        pred_dir = prediction_root / stage / str(args.variant)
        try:
            summary = train_scale_aware_dataset(
                replicate_dir,
                pred_dir,
                args,
                seed=int(args.seed) + int(dss.STAGE_ORDER.get(stage, 0)),
            )
            status_rows.append(
                {
                    "stage": stage,
                    "status": "completed",
                    "replicate_dir": str(replicate_dir),
                    "prediction_path": str(PROJECT_ROOT / summary["prediction_path"]),
                    "runtime_seconds": summary["runtime_seconds"],
                    "error": "",
                }
            )
        except Exception as exc:  # noqa: BLE001
            status_rows.append(
                {
                    "stage": stage,
                    "status": "failed",
                    "replicate_dir": str(replicate_dir),
                    "prediction_path": "",
                    "runtime_seconds": np.nan,
                    "error": repr(exc),
                }
            )
            if args.fail_fast:
                raise

    status = pd.DataFrame(status_rows)
    status.to_csv(dss_out / "stagewise_scale_candidate_status.csv", index=False)
    completed = status[status["prediction_path"].astype(str).ne("")]
    result = {"completed_stages": int(len(completed)), "failed_stages": int((status["status"] == "failed").sum())}
    summary_dir = dss_out / "summary"
    summary_dir.mkdir(exist_ok=True)
    if not completed.empty:
        lr_summary = dss.summarize_stage_lr(completed)
        lr_summary.to_csv(summary_dir / "stage_lr_summary_long.csv", index=False)
        trajectories = dss.summarize_trajectories(lr_summary)
        trajectories.to_csv(summary_dir / "stage_lr_trajectories.csv", index=False)
        story = dss.select_story_candidates(trajectories)
        story.to_csv(summary_dir / "stage_lr_story_candidates.csv", index=False)
        dss.write_story_markdown(story, dss_out / "stagewise_scale_candidate_story_summary.md")
        focus = trajectories[trajectories["lr_pair"].isin(FOCUS_DSS)].copy()
        focus.to_csv(summary_dir / "focus_dss_lr_trajectories.csv", index=False)
        result.update(
            {
                "stage_lr_trajectories": str((summary_dir / "stage_lr_trajectories.csv").relative_to(PROJECT_ROOT)),
                "story_candidates": str((summary_dir / "stage_lr_story_candidates.csv").relative_to(PROJECT_ROOT)),
                "focus_lr_trajectories": str((summary_dir / "focus_dss_lr_trajectories.csv").relative_to(PROJECT_ROOT)),
            }
        )
    (dss_out / "dss_reproduction_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["crc", "dss", "all"], default="all")
    parser.add_argument("--crc-dataset-dir", default=str(DEFAULT_CRC_DIR))
    parser.add_argument("--dss-replicate-root", default=str(DEFAULT_DSS_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--variant", default="mixture_ranklocal_responsibility")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--k-nn", type=int, default=12)
    parser.add_argument("--h-dim", type=int, default=64)
    parser.add_argument("--lr-dim", type=int, default=24)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-train-samples", type=int, default=12000)
    parser.add_argument("--score-batch-size", type=int, default=65536)
    parser.add_argument("--positive-quantile", type=float, default=0.9)
    parser.add_argument("--negative-quantile", type=float, default=0.2)
    parser.add_argument("--lr-cap-positive", type=int, default=50)
    parser.add_argument("--lr-cap-negative", type=int, default=200)
    parser.add_argument("--scale-target-weight", type=float, default=1.1)
    parser.add_argument("--diversity-weight", type=float, default=0.45)
    parser.add_argument("--responsibility-weight", type=float, default=0.35)
    parser.add_argument("--responsibility-margin", type=float, default=0.35)
    parser.add_argument("--responsibility-edges", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = {}
    if args.mode in {"crc", "all"}:
        outputs["crc"] = run_crc(args)
    if args.mode in {"dss", "all"}:
        outputs["dss"] = run_dss(args)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
